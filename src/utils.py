from pathlib import Path

import cv2
import numpy as np


def write_image_to_file(path: Path, image: np.ndarray):
    """
    cv2 文件名会编码错误!!!
    :param path:
    :param image:
    :return:
    """
    state, image_bytes = cv2.imencode(path.suffix, image)
    if not state:
        raise RuntimeError('Failed to encode image', path)
    else:
        path.write_bytes(image_bytes)
