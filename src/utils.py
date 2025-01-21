import base64
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
    if not image.size:
        return
    state, image_bytes = cv2.imencode(path.suffix, image)
    if not state:
        raise RuntimeError('Failed to encode image', path)
    else:
        path.write_bytes(image_bytes)


def create_webp_b64(image_path: Path) -> str:
    str_ = base64.b64encode(image_path.read_bytes()).decode()
    return f'data:image/webp;base64,{str_}'


def thumbnail_image(image: np.ndarray, target_width: int | None, target_height: int | None) -> np.ndarray:
    """
    缩放图片
    :param image:
    :param target_width:
    :param target_height:
    :return:
    """
    height, width = image.shape[:2]

    check_width = (target_width is None) or (target_width >= width)
    check_height = (target_height is None) or (target_height >= height)
    if check_width and check_height:
        return image

    if target_height is None:
        scale = target_width / width
    elif target_width is None:
        scale = target_height / height
    elif width > height:
        scale = target_width / width
    else:
        scale = target_height / height

    return cv2.resize(image, dsize=None, fx=scale, fy=scale)
