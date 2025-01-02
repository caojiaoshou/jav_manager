import typing as t
from pathlib import Path

import av
import numpy as np
from nudenet import NudeDetector


class _Record(t.NamedTuple):
    score: float
    img: np.ndarray | None

    def __bool__(self) -> bool:
        return self.img is not None


nude_detector = NudeDetector()

target_class = [
    # "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    # "MALE_BREAST_EXPOSED",
    # "ANUS_EXPOSED",
    # "FEET_EXPOSED",
    # "BELLY_COVERED",
    # "FEET_COVERED",
    # "ARMPITS_COVERED",
    # "ARMPITS_EXPOSED",
    # "FACE_MALE",
    # "BELLY_EXPOSED",
    # "MALE_GENITALIA_EXPOSED",
    # "ANUS_COVERED",
    # "FEMALE_BREAST_COVERED",
    # "BUTTOCKS_COVERED",
]

p_for_test = 'D:\L5\[JAV] [Uncensored] FC2 PPV 1888207 [1080p]\FC2-PPV-1888207_1.mp4'


def detect_video_preview(video_path: Path) -> np.ndarray:
    target_mapper = {k: _Record(0, None) for k in target_class}
    with open(video_path, mode='rb') as io:
        try:
            # 防止文件损坏导致FFMPEG无法读取
            container = av.open(io)

            second_pass = 0

            for packet in container.demux(video=0):
                if packet.is_keyframe:
                    for frame in packet.decode():
                        image = frame.to_ndarray(format='bgr24')
                        detect_result_list = nude_detector.detect(image)
                        for result in detect_result_list:
                            if result['class'] not in target_mapper:
                                continue
                            elif result['score'] < 0.75:
                                continue
                            else:
                                if target_mapper[result['class']].score < result['score']:
                                    target_mapper[result['class']] = _Record(result['score'], image)

                if all(target_mapper.values()):
                    break

                second_pass += packet.duration * packet.time_base
            container.close()
        except Exception:
            return

    to_stack_list = [r.img for r in target_mapper.values() if r.img is not None]
    if to_stack_list:
        return np.vstack(to_stack_list)
    else:
        return None


if __name__ == '__main__':
    detect_video_preview(p_for_test)
