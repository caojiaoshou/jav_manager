import typing as t

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.file_index import MODEL_STORAGE, VIDEO_FILE_FOR_TEST

_MODEL_DIR = MODEL_STORAGE / 'insightface'
_MODEL_DIR.mkdir(exist_ok=True)
_ANALYSER = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                         root=_MODEL_DIR.absolute().__str__())
_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))


class BestFace(t.NamedTuple):
    left: int
    top: int
    right: int
    bottom: int
    pitch: float
    yaw: float
    roll: float

    @classmethod
    def from_insightface_record(cls, record: any) -> 'BestFace':
        return cls(
            *map(int, record.bbox),
            *map(float, record.pose)
        )

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class FaceAnalysisResult(t.NamedTuple):
    best: BestFace
    frame_index: int
    age: float
    face_serial: np.ndarray


def cos_similarity(map_: np.ndarray, target: np.ndarray) -> np.float64 | np.ndarray:
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


def min_max_scale(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


def find_best_face(image_seq: t.Sequence[np.ndarray]) -> FaceAnalysisResult:
    record_list: list[tuple[int, any]] = []
    for image_index, image_array in enumerate(image_seq):
        image_female_face_list = [f for f in _ANALYSER.get(image_array) if (f.sex == 'F') and (f.det_score > 0.6)]
        if image_female_face_list:
            record_list.append((image_index, image_female_face_list[0]))

    det_weight = np.array([r[1].det_score for r in record_list], dtype=np.float64)
    det_weight = min_max_scale(det_weight)

    age_array = np.array([r[1].age for r in record_list], dtype=np.float64)
    serial_array = np.array([r[1].normed_embedding for r in record_list], dtype=np.float64)
    pose_array = np.array([r[1].pose for r in record_list], dtype=np.float64)

    pose_weight = np.abs(pose_array)
    pose_weight[:, 2] = pose_weight[:, 2] * 0.5
    pose_weight = np.prod(pose_weight, axis=1) ** (1.0 / 3.0)
    pose_weight = min_max_scale(pose_weight)

    preview_weight = det_weight - pose_weight
    preview_sort = np.argsort(preview_weight)
    preview_index_in_middleware = int(preview_sort[-1])
    preview_index_in_input, preview_record = record_list[preview_index_in_middleware]
    age_value = (det_weight * age_array).sum() / det_weight.sum()

    all_serial_value = np.mean(serial_array, axis=0)
    sim_to_mean = cos_similarity(serial_array, all_serial_value)
    mask = np.where(sim_to_mean > sim_to_mean.mean() - sim_to_mean.std(), 1, 0)
    serial_weight = det_weight * mask
    all_serial_value_2 = (serial_array.transpose() * serial_weight).sum(axis=1) / serial_weight.sum()

    preview_face = BestFace.from_insightface_record(preview_record)
    result = FaceAnalysisResult(best=preview_face, frame_index=preview_index_in_input, age=age_value,
                                face_serial=all_serial_value_2)
    return result


def crop_face_into_square(image: np.ndarray, face: BestFace) -> np.ndarray:
    box_center_x_y = ((face.left + face.right) // 2, (face.top + face.bottom) // 2)
    matrix_for_rotation = cv2.getRotationMatrix2D(box_center_x_y, face.roll / 3 * 2, 1.0)
    rotated_img = cv2.warpAffine(image, matrix_for_rotation, (image.shape[1], image.shape[0]))

    new_x1 = int(max(face.left - face.width // 2, 0))
    new_y1 = int(max(face.top - face.height // 2, 0))
    new_x2 = int(max(face.right + face.width // 2, 0))
    new_y2 = int(max(face.bottom + face.height // 2, 0))
    new_img = rotated_img[new_y1: new_y2, new_x1:new_x2]
    return new_img


def _test_crop():
    from src.file_index import IMAGE_FILE_FOR_TEST
    img = cv2.imread(IMAGE_FILE_FOR_TEST.absolute().__str__())

    face = BestFace.from_insightface_record(_ANALYSER.get(img)[0])

    crop_image = crop_face_into_square(img, face)

    cv2.imshow('', crop_image)
    cv2.waitKey(0)


def _test_detect():
    from src.loader import iter_keyframe_bgr24

    keyframe_records = list(iter_keyframe_bgr24(VIDEO_FILE_FOR_TEST))
    out = find_best_face([f.bgr_array for f in keyframe_records])

    cv2.imshow('', keyframe_records[out.frame_index].bgr_array)
    cv2.waitKey(0)


if __name__ == '__main__':
    _test_crop()
