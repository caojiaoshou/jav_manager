import typing as t
from pathlib import Path

import av
import numpy as np

from src.scene import FrameRecord


def get_audio_samples_as_float32_array(file_path, sample_rate=16000, mono=True) -> np.ndarray:
    with av.open(file_path) as container:
        audio_stream = [stream for stream in container.streams if stream.type == 'audio'][0]

        resampler = av.AudioResampler(format='s16', layout='mono' if mono else audio_stream.layout.name,
                                      rate=sample_rate)

        samples = []
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                # Resample the frame to the desired sample rate and format
                frame = resampler.resample(frame)
                # Convert the frame to a NumPy array
                array = frame[0].to_ndarray()
                samples.append(array)

    samples_np = np.hstack(samples).astype(np.float32)
    return samples_np


def iter_frame_bgr24(video_path: Path) -> t.Generator[FrameRecord, None | bool, None]:
    with open(video_path, mode='rb') as io:
        try:
            # 防止文件损坏导致FFMPEG无法读取
            container = av.open(io)

            second_pass = 0

            for packet in container.demux(video=0):
                second_pass += packet.duration * packet.time_base
                if packet.is_keyframe:
                    if packet_frame_list := packet.decode():
                        stop_trigger = yield FrameRecord(
                            float(second_pass),
                            packet_frame_list[0].to_ndarray(format='bgr24')
                        )
                        if stop_trigger:
                            break
            container.close()
        finally:
            ...
