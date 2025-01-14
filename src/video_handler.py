import gc
import itertools
import pathlib
import time
import typing as t
from heapq import nlargest

import cv2
import numpy as np

from src.body_part import process_frame_for_detections, BodyPartDetectionCollection
from src.dao import VideoFace, VideoScene, VideoBodyPart
from src.face import crop_and_rotate_face_into_square, select_best_female_face, FaceNotFoundError
from src.file_index import VIDEO_FILE_FOR_TEST, TEMP_STORAGE
from src.loader import iter_keyframe_bgr24, pack_for_360p_webm, parse_frame_ts, calculate_frame_ts, FrameRecord, \
    extract_frame_ts
from src.logger import configure_logger
from src.rmbg import get_foreground_mask
from src.scene import create_frame_diff, FrameDiffRecord

_LOGGER = configure_logger('video')


class VideoFullWorkResult(t.NamedTuple):
    faces: list[VideoFace]
    quick_look: bytes
    scenes: list[VideoScene]
    body_parts: list[VideoBodyPart]


def video_full_work(p: pathlib.Path) -> VideoFullWorkResult:
    start_at = time.time()
    _LOGGER.debug(f'提取关键帧 {p}')

    # 有些视频一分钟120关键帧.普通质量一般2-3秒一个关键帧
    original_key_frame_count = 0
    preview_start_at = -128  # 保留第一帧
    keyframe_record_list = []
    for keyframe_record in iter_keyframe_bgr24(p):
        original_key_frame_count += 1
        if keyframe_record.start_at > preview_start_at + 1.4:
            preview_start_at = keyframe_record.start_at
            keyframe_record_list.append(keyframe_record)
    _LOGGER.debug(
        f'提取关键帧 用时 {time.time() - start_at:.2f}s, 找到 {original_key_frame_count} 帧,保留 {len(keyframe_record_list)} 帧')

    start_at = time.time()
    _LOGGER.debug(f'识别身体部位 {p}')
    keyframe_detection_list = [
        process_frame_for_detections(keyframe_record.bgr_array)
        for keyframe_record in keyframe_record_list
    ]
    _LOGGER.debug(f'识别身体部位 用时 {time.time() - start_at:.2f}s')

    start_at = time.time()
    _LOGGER.debug(f'计算视频帧差值 {p}')
    diff_list = create_frame_diff(keyframe_record_list)
    _LOGGER.debug(f'计算视频帧差值 用时 {time.time() - start_at:.2f}s')

    composite_list: list[tuple[FrameRecord, BodyPartDetectionCollection, FrameDiffRecord]] = [
        (i, j, k)
        for i, j, k in zip(keyframe_record_list[1:], keyframe_detection_list[1:], diff_list)
    ]

    # 处理预览帧
    base_upper = nlargest(1, composite_list, key=lambda tup: tup[1].upper_body)
    base_lower = nlargest(1, composite_list, key=lambda tup: tup[1].lower_body)
    base_oral = nlargest(2, composite_list, key=lambda tup: tup[1].oral_penetration.confidence)
    base_vaginal = nlargest(2, composite_list, key=lambda tup: tup[1].vaginal_penetration.confidence)
    base_diff = nlargest(2, composite_list, key=lambda tup: tup[2].iqr_z_score)
    concat_base = list(itertools.chain(base_upper, base_lower, base_diff, base_oral, base_vaginal))

    concat_base_start_at = sorted({i[0].start_at for i in concat_base})

    # 处理预览时间轴
    scenes_ts_image = [
        VideoScene(record.start_at, record.bgr_array)
        for record in keyframe_record_list
        if record.start_at in concat_base_start_at
    ]

    # 处理面部识别
    start_at = time.time()
    _LOGGER.debug(f'识别面部 {p}')
    prob_female_face_frames = [
        record_tuple[0].bgr_array
        for record_tuple in composite_list
        if record_tuple[1].face.confidence >= 0.7
    ]
    face_seq = []
    try:
        best_female_face = select_best_female_face(prob_female_face_frames)
        face_target_frame = prob_female_face_frames[best_female_face.frame_index]
        face_crop_image = crop_and_rotate_face_into_square(
            np.dstack([face_target_frame, get_foreground_mask(face_target_frame)]),
            best_female_face.best
        )
        _LOGGER.debug(f'识别面部 用时 {time.time() - start_at:.2f}s')
        face = VideoFace(best_female_face.face_serial, best_female_face.age, face_target_frame, face_crop_image)
        face_seq.append(face)
    except FaceNotFoundError:
        _LOGGER.debug(f'识别面部 用时 {time.time() - start_at:.2f}s')
        _LOGGER.warning(f'未识别到女性面部 {p}')
    # 处理身体部位
    body_parts = []
    for part in ['butt', 'breast', 'pussy', 'feet', 'bar']:
        frame_record, *_ = nlargest(1, composite_list, key=lambda tup: tup[1].__getattribute__(part).confidence)[0]
        body_parts.append(VideoBodyPart(part, frame_record.start_at, frame_record.bgr_array))

    # 调整处理顺序,并确保释放内存,看看能不能减轻内存占用
    del keyframe_record_list
    del composite_list
    gc.collect()

    # 处理预览视频
    start_at = time.time()
    _LOGGER.debug(f'生成预览视频 {p}')
    all_frame_ts = parse_frame_ts(p)
    quick_look_slicers = [
        calculate_frame_ts(all_frame_ts, start_at, 1.5)
        for start_at in concat_base_start_at
    ]

    # 处理高帧率问题
    quick_look_frames = []
    total_extract_frame_count = 0
    for slicer in quick_look_slicers:
        previous_start_at = -128  # 保留第一帧
        for frame in extract_frame_ts(p, slicer):
            total_extract_frame_count += 1
            if frame.start_at > previous_start_at + 0.030:  # 这是一个取巧计算 fps30是0.033一帧. fps60是0.017. 这样大概可以确保实际帧率约为30fps
                previous_start_at = frame.start_at
                quick_look_frames.append(frame)
    _LOGGER.debug(f'总共提取 {total_extract_frame_count} 帧, 保留 {len(quick_look_frames)} 帧')
    quick_look_video_bytes = pack_for_360p_webm(quick_look_frames)
    _LOGGER.debug(f'生成预览视频 用时 {time.time() - start_at:.2f}s')

    return VideoFullWorkResult(face_seq, quick_look_video_bytes, scenes_ts_image, body_parts)


if __name__ == '__main__':
    result = video_full_work(VIDEO_FILE_FOR_TEST)

    video_path = TEMP_STORAGE / VIDEO_FILE_FOR_TEST.with_suffix('.webm').name
    video_path.write_bytes(result.quick_look)

    image_path = TEMP_STORAGE / VIDEO_FILE_FOR_TEST.with_suffix('.webp').name
    cv2.imwrite(image_path.absolute().__str__(), result.faces[0].crop_image)

    for scene_record in result.scenes:
        cv2.imwrite(TEMP_STORAGE / f'scene_{scene_record.start_at:.2f}.webp', scene_record.frame)

    for body_record in result.body_parts:
        cv2.imwrite(TEMP_STORAGE / f'body_{body_record.part}.webp', body_record.frame)
