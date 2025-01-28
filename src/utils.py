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


def calculate_cos_similarity(map_: np.ndarray, target: np.ndarray) -> np.float64 | np.ndarray:
    # 如果输入是一维数组，则增加一个轴以支持广播机制
    if map_.ndim == 1:
        map_ = map_[np.newaxis, :]
    if target.ndim == 1:
        target = target[np.newaxis, :]

    # 计算两个向量的L2范数（欧几里得范数）
    map_norm = np.linalg.norm(map_, axis=1)
    target_norm = np.linalg.norm(target, axis=1)

    # 计算点积并利用广播机制来匹配形状
    dot_product = np.sum(map_ * target, axis=1)

    # 计算并返回余弦相似度
    similarity = dot_product / (map_norm * target_norm)

    # 如果输入是单个向量，则返回标量值；否则返回相似度数组
    if len(similarity) == 1:
        return similarity[0]
    else:
        return similarity
