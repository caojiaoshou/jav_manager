import statistics
import time
import typing as t
from pathlib import Path

import cv2
import numpy as np
from nudenet import NudeDetector

from loader import iter_frame_bgr24


class _Record(t.NamedTuple):
    score: float
    img: np.ndarray | None

    def __bool__(self) -> bool:
        return self.img is not None


nude_detector = NudeDetector()

target_class = [
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]

p_for_test = 'D:\L5\[JAV] [Uncensored] FC2 PPV 1888207 [1080p]\FC2-PPV-1888207_1.mp4'


def detect_video_preview(video_path: Path) -> np.ndarray:
    detect_cost_list = []
    decode_cost_list = []
    target_mapper = {k: _Record(0, None) for k in target_class}
    frame_generator = iter_frame_bgr24(video_path)
    while not all(target_mapper.values()):
        try:
            decode_start_at = time.time()
            frame_record = next(frame_generator)
            decode_cost_list.append(time.time() - decode_start_at)
        except StopIteration:
            break
        detect_start_at = time.time()
        detect_result_list = nude_detector.detect(frame_record.bgr_array)
        detect_cost_list.append(time.time() - detect_start_at)
        for result in detect_result_list:
            if result['class'] not in target_mapper:
                continue
            elif result['score'] < 0.75:
                continue
            else:
                if target_mapper[result['class']].score < result['score']:
                    target_mapper[result['class']] = _Record(result['score'], frame_record.bgr_array)
    print(f'{statistics.mean(detect_cost_list)=}, {statistics.stdev(detect_cost_list)=}, {len(detect_cost_list)=}')
    print(f'{statistics.mean(decode_cost_list)=}, {statistics.stdev(decode_cost_list)=},{len(decode_cost_list)=}')
    to_stack_list = [r.img for r in target_mapper.values() if r.img is not None]
    if to_stack_list:
        return np.vstack(to_stack_list)
    else:
        return None


if __name__ == '__main__':
    cv2.imshow('', cv2.resize(detect_video_preview(p_for_test), (640, 480)))
    cv2.waitKey(0)
