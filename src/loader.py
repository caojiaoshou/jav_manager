import io
import typing as t
from pathlib import Path

import av
import cv2
import numpy as np

from src.file_index import VIDEO_FILE_FOR_TEST, TEMP_STORAGE


def get_audio_samples_as_float32_array(file_path, sample_rate=16000, mono=True) -> np.ndarray:
    with av.open(file_path) as container:
        audio_stream = [stream for stream in container.streams if stream.type == 'audio'][0]

        resampler = av.AudioResampler(format='s16', layout='mono' if mono else audio_stream.layout.name,
                                      rate=sample_rate)

        samples = []
        for packet in container.demux(audio_stream):
            try:
                for frame in packet.decode():
                    # Resample the frame to the desired sample rate and format
                    frame = resampler.resample(frame)
                    # Convert the frame to a NumPy array
                    array = frame[0].to_ndarray()
                    samples.append(array)
            except Exception as e:
                break
    samples_np = np.hstack(samples).astype(np.float32)
    return samples_np


class FrameRecord(t.NamedTuple):
    start_at: float
    bgr_array: np.ndarray


def iter_keyframe_bgr24(video_path: Path) -> t.Generator[FrameRecord, None | bool, None]:
    with av.open(video_path) as container:
        # 防止文件损坏导致FFMPEG无法读取
        second_pass = 0

        for packet in container.demux(video=0):
            if packet.is_keyframe:
                if packet_frame_list := packet.decode():
                    yield FrameRecord(
                        float(second_pass),
                        packet_frame_list[0].to_ndarray(format='bgr24')
                    )
            second_pass += packet.duration * packet.time_base


class FrameTs(t.NamedTuple):
    packet_index: int
    start_at: float
    is_keyframe: bool


def parse_frame_ts(video_path: Path) -> list[FrameTs]:
    res_list = []
    with av.open(video_path) as container:
        second_pass = 0
        for frame_index, packet in enumerate(container.demux(video=0)):
            res_list.append(FrameTs(frame_index, float(second_pass), is_keyframe=packet.is_keyframe))
            second_pass += packet.duration * packet.time_base
    return res_list


class Slicer(t.NamedTuple):
    keyframe_index: int
    start_at_index: int
    end_at_index: int

    @property
    def decode_size(self) -> int:
        return self.end_at_index - self.keyframe_index + 1

    @property
    def data_size(self) -> int:
        return self.end_at_index - self.start_at_index + 1


def calculate_frame_ts(ls: list[FrameTs], start_at_ts: float, duration: float) -> Slicer:
    end_at_ts = start_at_ts + duration
    keyframe_index = 0
    start_at_frame = 0
    end_at_frame = 0
    for f in ls:
        if f.is_keyframe and f.start_at < start_at_ts:
            keyframe_index = f.packet_index
        elif f.start_at <= start_at_ts:
            start_at_frame = f.packet_index
        elif f.start_at <= end_at_ts:
            end_at_frame = f.packet_index
        else:
            break
    return Slicer(keyframe_index, start_at_frame, end_at_frame)


def extract_frame_ts(video_path: Path, slicer: Slicer) -> t.Generator[FrameRecord, None, None]:
    with av.open(video_path) as container:
        frame_start_at = 0
        for index, packet in enumerate(container.demux(video=0)):
            if index >= slicer.keyframe_index:
                packet_frame_list = packet.decode()
                if (index >= slicer.start_at_index) and packet_frame_list:
                    f = packet_frame_list[0].to_ndarray(format='bgr24')
                    yield FrameRecord(float(frame_start_at), f)
                if index >= slicer.end_at_index:
                    break
            frame_start_at += packet.duration * packet.time_base


def resize_and_crop(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    调整图像大小至指定宽高，并保持宽高比例，通过居中裁剪适应新尺寸。

    :param image: 输入的图像（numpy array）
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :return: 调整后并裁剪的图像
    """
    original_height, original_width = image.shape[:2]

    # 计算缩放因子
    ratio_width = target_width / float(original_width)
    ratio_height = target_height / float(original_height)

    # 选择最小的缩放因子以保证图像完全包含在目标尺寸内
    ratio = min(ratio_width, ratio_height)

    # 使用计算得到的缩放因子调整图像大小
    new_size = (int(original_width * ratio), int(original_height * ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # 获取调整后的尺寸
    resized_height, resized_width = resized_image.shape[:2]

    # 计算裁剪区域
    start_x = max(0, (resized_width - target_width) // 2)
    start_y = max(0, (resized_height - target_height) // 2)

    # 居中裁剪
    cropped_image = resized_image[start_y:start_y + target_height, start_x:start_x + target_width]

    return cropped_image


def pack_for_360p_webm(frame_record_list: list[FrameRecord]) -> bytes:
    with io.BytesIO() as buffer:
        with av.open(buffer, mode='w', format='webm') as container:
            stream = container.add_stream('libvpx', 30)
            stream.width = 640
            stream.height = 360
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '10'}
            for image_r in frame_record_list:
                img = resize_and_crop(image_r.bgr_array, 640, 360)
                frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            packet = stream.encode(None)
            if packet is not None:
                container.mux(packet)
        return buffer.getvalue()


def get_video_duration(video_path: str) -> float:
    try:
        container = av.open(video_path)
        duration = container.duration / av.time_base
        return float(duration)
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0.0


if __name__ == '__main__':
    frame_ts_list = parse_frame_ts(VIDEO_FILE_FOR_TEST)
    ls = calculate_frame_ts(frame_ts_list, start_at_ts=18 * 60 + 1, duration=1.2)
    frame_record_list = list(extract_frame_ts(VIDEO_FILE_FOR_TEST, ls))
    b = pack_for_360p_webm(frame_record_list)
    (TEMP_STORAGE / 'test.webm').write_bytes(b)
