import fractions
import io
import time
import typing as t
from pathlib import Path

import av
import cv2
import numpy as np

from src.file_index import VIDEO_FILE_FOR_TEST, TEMP_STORAGE

_cuda_option = {'hwaccel': 'cuvid', 'hwaccel_output_format': 'cuda'}


def insert_silence(samples, duration_in_seconds, sample_rate=16000):
    """Insert silence into the audio samples."""
    silence_length = int(duration_in_seconds * sample_rate)
    silence = np.zeros(silence_length, dtype=np.float32)
    return np.concatenate([samples, silence])


def _create_silence(duration_in_seconds, sample_rate=16000):
    """Insert silence into the audio samples."""
    silence_length = int(duration_in_seconds * sample_rate)
    silence = np.zeros(silence_length, dtype=np.float32)
    return silence


def get_audio_samples_as_float32_array(file_path, sample_rate=16000, mono=True) -> np.ndarray:
    samples = []
    last_pts = None
    with av.open(file_path) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == 'audio'), None)
        if not audio_stream:
            raise ValueError("No audio stream found in the file.")

        resampler = av.AudioResampler(format='s16', layout='mono' if mono else audio_stream.layout.name,
                                      rate=sample_rate)

        for packet in container.demux(audio_stream):
            try:
                for frame in packet.decode():
                    # Resample the frame to the desired sample rate and format
                    frame = resampler.resample(frame)
                    # Convert the frame to a NumPy array
                    array = frame[0].to_ndarray()
                    samples.append(array)

            except av.error.InvalidDataError as e:
                print(f"Warning: Invalid data encountered at timestamp {packet.pts}. Skipping this packet.")
                if last_pts is not None:
                    # Calculate the time difference between the last valid PTS and the current packet's PTS
                    time_diff = (packet.pts - last_pts) * audio_stream.time_base
                    samples.append(_create_silence(time_diff, sample_rate=sample_rate))
                continue
            finally:
                last_pts = packet.pts
    samples_np = np.hstack(samples).astype(np.float32)
    return samples_np


class FrameRecord(t.NamedTuple):
    start_at: float
    bgr_array: np.ndarray


def _core_iter_packet(video_path: Path, cuda: bool = False) -> t.Generator[
    tuple[int, fractions.Fraction, av.Packet], None, None]:
    """
    抽取方法 对时间轴做特殊处理.有些视频packet没有duration.所以要推测时间
    :return packet_index, start_at, packet
    """
    options = {} if not cuda else _cuda_option
    with av.open(video_path, options=options) as container:
        second_pass_dec = fractions.Fraction(0)
        # 有些视频packet里边没有保存duration.取平均值
        video_stream = container.streams.video[0]
        avg_frame_duration = video_stream.time_base * video_stream.duration / video_stream.frames
        for index, packet in enumerate(container.demux(video=0)):
            yield index, second_pass_dec, packet
            if packet.duration:
                second_pass_dec += packet.duration * packet.time_base
            else:
                second_pass_dec += avg_frame_duration


def iter_keyframe_bgr24(video_path: Path, cuda: bool = False) -> t.Generator[FrameRecord, None | bool, None]:
    for index, start_at, packet in _core_iter_packet(video_path, cuda):
        if packet.is_keyframe:
            # 这里返回的是列表.观测结果实际只有0或者1个帧
            if packet_frame_list := packet.decode():
                yield FrameRecord(
                    float(start_at),
                    packet_frame_list[0].to_ndarray(format='bgr24')
                )


class FrameTs(t.NamedTuple):
    packet_index: int
    start_at: float
    is_keyframe: bool


def parse_frame_ts(video_path: Path) -> list[FrameTs]:
    return [
        FrameTs(frame_index, float(second_pass), packet.is_keyframe)
        for frame_index, second_pass, packet in _core_iter_packet(video_path)
    ]


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


def extract_frame_ts(video_path: Path, slicer: Slicer, cuda: bool = False) -> t.Generator[FrameRecord, None, None]:
    for index, start_at, packet in _core_iter_packet(video_path, cuda):
        last_frame_decoded = None
        if index >= slicer.keyframe_index:
            if index > slicer.end_at_index:
                break
            try:
                packet_frame_list = packet.decode()
                if (index >= slicer.start_at_index) and packet_frame_list:
                    f = packet_frame_list[0].to_ndarray(format='bgr24')
                    last_frame_decoded = f
                    yield FrameRecord(float(start_at), f)
            except av.error.InvalidDataError as e:
                print(f"Warning: Invalid data encountered at timestamp {packet.pts}. Skipping this packet.")
                if not last_frame_decoded:
                    continue
                else:
                    yield FrameRecord(float(start_at), last_frame_decoded)


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
    """
    自动控制帧率
    :param frame_record_list:
    :return:
    """
    # 统计帧率, 由于实务有多重帧率的视频,而且视频是非连续切片.所以要推算帧率进行压缩
    frame_start_at_array = np.array([f.start_at for f in frame_record_list], dtype=np.float32)
    frame_diff = frame_start_at_array[1:] - frame_start_at_array[:-1]
    q25 = np.quantile(frame_diff, 0.25)
    q75 = np.quantile(frame_diff, 0.75)
    lb = q25 - 1.5 * (q75 - q25)
    ub = q75 + 1.5 * (q75 - q25)
    frame_diff_2 = frame_diff[(frame_diff >= lb - 1e-5) & (frame_diff <= ub + 1e-5)]
    fps = int(np.round(1.0 / np.mean(frame_diff_2), 0))

    with io.BytesIO() as buffer:
        with av.open(buffer, mode='w', format='webm') as container:
            stream = container.add_stream('libvpx', fps)
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


def _test_extract_and_pack():
    frame_ts_list = parse_frame_ts(VIDEO_FILE_FOR_TEST)
    ls = calculate_frame_ts(frame_ts_list, start_at_ts=18 * 60 + 1, duration=12)
    frame_record_list = list(extract_frame_ts(VIDEO_FILE_FOR_TEST, ls))
    b = pack_for_360p_webm(frame_record_list)
    (TEMP_STORAGE / 'test.webm').write_bytes(b)


def _test_cuda():
    cuda_start_at = time.time()
    list(iter_keyframe_bgr24(VIDEO_FILE_FOR_TEST, cuda=True))
    cuda_cost = time.time() - cuda_start_at

    cpu_start_at = time.time()
    list(iter_keyframe_bgr24(VIDEO_FILE_FOR_TEST, cuda=False))
    cpu_cost = time.time() - cpu_start_at

    print(f'key frame test, cuda cost: {cuda_cost:.2f}, cpu cost: {cpu_cost:.2f}')

    frame_ts_list = parse_frame_ts(VIDEO_FILE_FOR_TEST)
    ls = calculate_frame_ts(frame_ts_list, start_at_ts=18 * 60 + 1, duration=15)

    cuda_start_at = time.time()
    frame_record_list = list(extract_frame_ts(VIDEO_FILE_FOR_TEST, ls, True))
    cuda_cost = time.time() - cuda_start_at

    cpu_start_at = time.time()
    frame_record_list = list(extract_frame_ts(VIDEO_FILE_FOR_TEST, ls, False))
    cpu_cost = time.time() - cpu_start_at

    print(f'frame test, cuda cost: {cuda_cost:.2f}, cpu cost: {cpu_cost:.2f}')


def _find_decoders():
    for i in av.codecs_available:
        print(i)


def _test_audio():
    import pickle
    from src.file_index import TEMP_STORAGE
    sample_rate = 16000
    start_at = time.time()
    result = get_audio_samples_as_float32_array(VIDEO_FILE_FOR_TEST, sample_rate)
    cost = time.time() - start_at
    print(f'解码 cost: {cost:.2f}, len: {result.shape[1] / sample_rate / 60:.2f}分钟')

    pickle_start_at = time.time()
    p = TEMP_STORAGE / 'test.pkl'
    with open(p, 'wb') as io:
        pickle.dump(result, io)
    with open(p, 'rb') as io:
        pickle.load(io)
    print(f'pickle cost: {time.time() - pickle_start_at:.2f}, size:{p.stat().st_size / 1024 / 1024:.2f}MB')
    p.unlink()


if __name__ == '__main__':
    _test_audio()
