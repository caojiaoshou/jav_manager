import dataclasses
import functools
import statistics
import typing as t
from pathlib import Path

import cv2
import numpy as np
from nudenet import NudeDetector

from loader import iter_keyframe_bgr24, FrameRecord
from src.file_index import VIDEO_FILE_FOR_TEST


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


@dataclasses.dataclass(frozen=True)
class DetectionResult:
    confidence: float = 0
    bbox: t.Tuple[int, int, int, int] | None = None

    @property
    def confidence_en5(self) -> float:
        return self.confidence + 1e-5

    @property
    def box_size(self) -> int:
        if self.bbox is None:
            return 0
        else:
            return self.bbox[2] * self.bbox[3]


@dataclasses.dataclass
class BodyPartDetectionCollection:
    face: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    butt: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    breast: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    pussy: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    penis: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    feet: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    bar: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    leg: DetectionResult = dataclasses.field(default_factory=DetectionResult)
    pans: DetectionResult = dataclasses.field(default_factory=DetectionResult)

    @staticmethod
    def _calculate_overlap(dt_1: DetectionResult, dt_2: DetectionResult) -> DetectionResult:
        pct = 0
        bbox_target = None
        if dt_1.confidence and dt_2.confidence:
            overlap_ratio = calculate_overlap_ratio(dt_1.bbox, dt_2.bbox)
            if overlap_ratio > 0:
                pct = statistics.geometric_mean([dt_1.confidence, dt_2.confidence, overlap_ratio])
                bbox_target = calculate_enclosing_rectangle(dt_1.bbox, dt_2.bbox)

        return DetectionResult(pct, bbox_target)

    @functools.cached_property
    def oral_penetration(self) -> DetectionResult:
        return self._calculate_overlap(self.face, self.penis)

    @functools.cached_property
    def vaginal_penetration(self) -> DetectionResult:
        return self._calculate_overlap(self.pussy, self.penis)

    @functools.cached_property
    def upper_body(self) -> float:
        return statistics.geometric_mean(
            (
                self.face.confidence_en5,
                max(self.bar.confidence_en5, self.breast.confidence_en5)
            )
        )

    @functools.cached_property
    def lower_body(self) -> float:
        return statistics.geometric_mean(
            (
                self.leg.confidence_en5,
                max(self.butt.confidence_en5, self.pans.confidence_en5)
            )
        )


nude_detector = NudeDetector()


def process_frame_for_detections(frame: np.ndarray) -> BodyPartDetectionCollection:
    _record = BodyPartDetectionCollection()
    detect_result_list = nude_detector.detect(frame)
    for detect_result in detect_result_list:
        detection = DetectionResult(detect_result['score'], tuple(detect_result['box']))
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
            case 'FEMALE_BREAST_COVERED':
                _record.bar = detection
            case 'FEET_EXPOSED':
                _record.leg = detection
            case 'BUTTOCKS_COVERED':
                _record.pans = detection
    return _record


def process_video_for_detections(video_path: Path) -> list[tuple[FrameRecord, BodyPartDetectionCollection]]:
    res_list = []
    frame_generator = iter_keyframe_bgr24(video_path)
    while True:
        try:
            frame_record = next(frame_generator)
        except StopIteration:
            break

        detect_record = process_frame_for_detections(frame_record.bgr_array)
        res_list.append((frame_record, detect_record))

    return res_list


def _test():
    result = process_video_for_detections(VIDEO_FILE_FOR_TEST)
    result_for_oral_sorted = sorted(result, key=lambda x: x[1].oral_penetration.confidence, reverse=True)
    for result_for_oral in result_for_oral_sorted[0:5]:
        cv2.imshow('', result_for_oral[0].bgr_array)
        cv2.waitKey(0)
    result_for_vaginal_sorted = sorted(result, key=lambda x: x[1].vaginal_penetration.confidence, reverse=True)
    for result_for_vaginal in result_for_vaginal_sorted[0:5]:
        cv2.imshow('', result_for_vaginal[0].bgr_array)
        cv2.waitKey(0)


if __name__ == '__main__':
    _test()
