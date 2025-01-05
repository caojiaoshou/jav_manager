import pathlib

from src.body_part import process_frame_for_detections, BodyPartDetectionCollection
from src.file_index import VIDEO_FILE_FOR_TEST, TEMP_STORAGE
from src.loader import iter_keyframe_bgr24, pack_for_360p_webm, parse_frame_ts, calculate_frame_ts, extract_frame_ts, \
    FrameRecord
from src.scene import create_frame_diff, FrameDiffRecord


def create_video_website_style_webm_preview(p: pathlib.Path) -> bytes:
    keyframe_record_list = list(iter_keyframe_bgr24(p))

    keyframe_detection_list = [
        process_frame_for_detections(keyframe_record.bgr_array)
        for keyframe_record in keyframe_record_list
    ]

    diff_list = create_frame_diff(keyframe_record_list)

    composite_list: list[tuple[FrameRecord, BodyPartDetectionCollection, FrameDiffRecord]] = [
        (i, j, k)
        for i, j, k in zip(keyframe_record_list[1:], keyframe_detection_list[1:], diff_list)
    ]

    base_upper = sorted(composite_list, key=lambda tup: tup[1].upper_body, reverse=True)
    base_lower = sorted(composite_list, key=lambda tup: tup[1].lower_body, reverse=True)
    base_oral = sorted(composite_list, key=lambda tup: tup[1].oral_penetration.confidence, reverse=True)
    base_vaginal = sorted(composite_list, key=lambda tup: tup[1].vaginal_penetration.confidence, reverse=True)
    base_diff = sorted(composite_list, key=lambda tup: tup[2].iqr_z_score, reverse=True)

    concat_frame_list = [
        base_upper[0],
        base_lower[0],
        *base_diff[:2],
        *base_oral[:2],
        *base_vaginal[:2]
    ]
    concat_frame_start_at = sorted({i[0].start_at for i in concat_frame_list})

    all_frame_index = parse_frame_ts(p)

    slicers = [
        calculate_frame_ts(all_frame_index, start_at, 1.5) for start_at in concat_frame_start_at
    ]

    images = []
    for part in slicers:
        images.extend(extract_frame_ts(p, part))

    preview_video = pack_for_360p_webm(images)

    return preview_video


if __name__ == '__main__':
    out_path = TEMP_STORAGE / VIDEO_FILE_FOR_TEST.with_suffix('.webm').name
    webm_bytes = create_video_website_style_webm_preview(VIDEO_FILE_FOR_TEST)
    out_path.write_bytes(webm_bytes)
