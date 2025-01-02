import dataclasses
import functools
import statistics
import time
import typing as t
from pathlib import Path

import cv2
import numpy as np
from nudenet import NudeDetector

from loader import iter_frame_bgr24, FrameRecord


def calculate_overlap_ratio(bbox1, bbox2) -> float:
    # 解包bbox tuple (left, top, width, height)
    left1, top1, w1, h1 = bbox1
    left2, top2, w2, h2 = bbox2

    # 计算每个bbox的边框
    right1, bottom1 = left1 + w1, top1 + h1
    right2, bottom2 = left2 + w2, top2 + h2

    # 计算重叠区域的边框
    overlap_left = max(left1, left2)
    overlap_right = min(right1, right2)
    overlap_top = max(top1, top2)
    overlap_bottom = min(bottom1, bottom2)

    # 检查是否有重叠
    if overlap_left < overlap_right and overlap_top < overlap_bottom:
        # 如果有重叠，计算重叠面积
        overlap_width = overlap_right - overlap_left
        overlap_height = overlap_bottom - overlap_top
        overlap_area = overlap_width * overlap_height
    else:
        # 如果没有重叠，重叠面积为0
        overlap_area = 0

    # 计算两个bbox的面积
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算并集面积（总面积）
    union_area = area1 + area2 - overlap_area

    # 计算重叠比例
    if union_area > 0:
        overlap_ratio = overlap_area / union_area
    else:
        overlap_ratio = 0  # 防止除以零的情况

    return overlap_ratio


def calculate_enclosing_rectangle(bbox1, bbox2) -> tuple[float, float, float, float]:
    # 解包bbox tuple (left, top, width, height)
    left1, top1, w1, h1 = bbox1
    left2, top2, w2, h2 = bbox2

    # 计算每个bbox的边框
    right1, bottom1 = left1 + w1, top1 + h1
    right2, bottom2 = left2 + w2, top2 + h2

    # 计算最小外接矩形的边框
    enclosing_left = min(left1, left2)
    enclosing_right = max(right1, right2)
    enclosing_top = min(top1, top2)
    enclosing_bottom = max(bottom1, bottom2)

    # 返回最小外接矩形作为一个tuple (left, top, width, height)
    return (
        enclosing_left,
        enclosing_top,
        enclosing_right - enclosing_left,
        enclosing_bottom - enclosing_top
    )


@dataclasses.dataclass
class _Detection:
    confidence: float = 0
    bbox: t.Tuple[int, int, int, int] | None = None


@dataclasses.dataclass
class _RecordV2:
    face: _Detection = dataclasses.field(default_factory=_Detection)
    butt: _Detection = dataclasses.field(default_factory=_Detection)
    breast: _Detection = dataclasses.field(default_factory=_Detection)
    pussy: _Detection = dataclasses.field(default_factory=_Detection)
    penis: _Detection = dataclasses.field(default_factory=_Detection)
    feet: _Detection = dataclasses.field(default_factory=_Detection)

    @staticmethod
    def _create_overlap(dt_1: _Detection, dt_2: _Detection) -> _Detection:
        pct = 0
        bbox_target = None
        if dt_1.confidence and dt_2.confidence:
            overlap_ratio = calculate_overlap_ratio(dt_1.bbox, dt_2.bbox)
            if overlap_ratio > 0:
                pct = statistics.geometric_mean([dt_1.confidence, dt_2.confidence, overlap_ratio])
                bbox_target = calculate_enclosing_rectangle(dt_1.bbox, dt_2.bbox)

        return _Detection(pct, bbox_target)

    @functools.cached_property
    def oral_penetration(self) -> _Detection:
        return self._create_overlap(self.face, self.penis)

    @functools.cached_property
    def vaginal_penetration(self) -> _Detection:
        return self._create_overlap(self.pussy, self.penis)


nude_detector = NudeDetector()


def frame_image_to_detection(frame: np.ndarray) -> _RecordV2:
    _record = _RecordV2()
    detect_result_list = nude_detector.detect(frame)
    for detect_result in detect_result_list:
        detection = _Detection(detect_result['score'], tuple(detect_result['box']))
        match detect_result['class']:
            case 'FACE_FEMALE':
                _record.face = detection
            case 'BUTTOCKS_EXPOSED':
                _record.butt = detection
            case 'FEMALE_GENITALIA_EXPOSED':
                _record.pussy = detection
            case 'MALE_GENITALIA_EXPOSED':
                _record.penis = detection
            case 'FEMALE_BREAST_EXPOSED':
                _record.breast = detection
            case 'FEET_EXPOSED':
                _record.feet = detection
    return _record


def detect_video_preview(video_path: Path) -> list[tuple[FrameRecord, _RecordV2]]:
    detect_cost_list = []
    decode_cost_list = []

    res_list = []
    frame_generator = iter_frame_bgr24(video_path)
    while True:
        try:
            decode_start_at = time.time()
            frame_record = next(frame_generator)
            decode_cost_list.append(time.time() - decode_start_at)
        except StopIteration:
            break
        detect_start_at = time.time()
        detect_record = frame_image_to_detection(frame_record.bgr_array)
        res_list.append((frame_record, detect_record))
        detect_cost_list.append(time.time() - detect_start_at)
    print(f'{statistics.mean(detect_cost_list)=}, {statistics.stdev(detect_cost_list)=}, {len(detect_cost_list)=}')
    print(f'{statistics.mean(decode_cost_list)=}, {statistics.stdev(decode_cost_list)=},{len(decode_cost_list)=}')
    return res_list


if __name__ == '__main__':
    p_for_test = 'D:\L5\[JAV] [Uncensored] FC2 PPV 1888207 [1080p]\FC2-PPV-1888207_1.mp4'
    result = detect_video_preview(p_for_test)
    result_for_oral_sorted = sorted(result, key=lambda x: x[1].oral_penetration.confidence, reverse=True)
    for result_for_oral in result_for_oral_sorted[0:5]:
        cv2.imshow('', result_for_oral[0].bgr_array)
        cv2.waitKey(0)
    result_for_vaginal_sorted = sorted(result, key=lambda x: x[1].vaginal_penetration.confidence, reverse=True)
    for result_for_vaginal in result_for_vaginal_sorted[0:5]:
        cv2.imshow('', result_for_vaginal[0].bgr_array)
        cv2.waitKey(0)
