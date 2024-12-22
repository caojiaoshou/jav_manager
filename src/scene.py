import typing as t

import cv2
import numpy as np

from src.loader import iter_frame_bgr24, FrameRecord

p_for_test = 'D:\L5\[JAV] [Uncensored] FC2 PPV 1888207 [1080p]\FC2-PPV-1888207_1.mp4'


def serial_image(bgr_array: np.ndarray) -> np.ndarray:
    scale_img = cv2.resize(bgr_array, (32, 32), interpolation=cv2.INTER_CUBIC)
    grey_img = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)

    dct_map = cv2.dct(np.float32(grey_img))
    dct_low_freq = dct_map[0:8, 0:8]
    threshold = np.mean(dct_low_freq)
    diff = dct_low_freq > threshold
    return diff.flatten()


class FrameDiffRecord(t.NamedTuple):
    seq_index: int
    iqr_z_score: float
    min_max: float


def create_frame_diff(frame_seq: t.Sequence[FrameRecord]) -> list[FrameDiffRecord]:
    serial_image_list = []

    for record in frame_seq:
        serial_image_list.append(serial_image(record.bgr_array))

    hash_array = np.array(serial_image_list)

    key_frame_diff = np.logical_xor(hash_array[0:-1, :], hash_array[1:, :])
    key_frame_diff = np.sum(key_frame_diff, axis=1)

    key_frame_iqr_z_score = (key_frame_diff - np.median(key_frame_diff)) / (
            np.quantile(key_frame_diff, 0.75) - np.quantile(key_frame_diff, 0.25))

    key_frame_min_max = (key_frame_diff - key_frame_diff.min()) / (key_frame_diff.max() - key_frame_diff.min())

    return [
        FrameDiffRecord(index + 1, *value) for index, value in enumerate(zip(key_frame_iqr_z_score, key_frame_min_max))
    ]


if __name__ == '__main__':
    frame_list = list(iter_frame_bgr24(p_for_test))
    scene_ = create_frame_diff(frame_list)

    for seq_index, iqr, mm in scene_:
        if iqr >= 1:
            rp = frame_list[seq_index - 1]
            cp = frame_list[seq_index]
            cv2.imwrite(f'../sample/{cp.sec_delta}_{iqr}.jpg', np.vstack([rp.bgr_array, cp.bgr_array]))
