import pathlib
import time
import typing as t

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.file_index import MODEL_STORAGE, VIDEO_FILE_FOR_TEST

_MODEL_DIR = MODEL_STORAGE / 'insightface'
_MODEL_DIR.mkdir(exist_ok=True)
_FACE_ANALYSER = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'],
                              root=_MODEL_DIR.absolute().__str__())
_FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))


class DetectedFace(t.NamedTuple):
    left: int
    top: int
    right: int
    bottom: int
    pitch: float
    yaw: float
    roll: float

    @classmethod
    def from_insightface_record(cls, record: any) -> 'DetectedFace':
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


class FaceDetectionResult(t.NamedTuple):
    best: DetectedFace
    frame_index: int
    age: float
    face_serial: np.ndarray


def _calculate_cos_similarity(map_: np.ndarray, target: np.ndarray) -> np.float64 | np.ndarray:
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


def _normalize_min_max(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min()) / (arr.max() - arr.min())


class FaceNotFoundError(Exception):
    ...


def select_best_female_face(image_sequence: t.Sequence[np.ndarray]) -> FaceDetectionResult:
    # 确保至少有一个图像
    if not image_sequence:
        raise FaceNotFoundError("Image sequence is empty")

    female_face_records = []
    for frame_index, frame_image in enumerate(image_sequence):
        detected_faces = _FACE_ANALYSER.get(frame_image)

        detected_female_faces = [
            f for f in detected_faces
            if (f.sex == 'F') and (f.det_score > 0.6)
        ]
        if detected_female_faces:
            female_face_records.append((frame_index, detected_female_faces[0]))

    if not female_face_records:
        raise FaceNotFoundError("No female faces detected")

    # 提取所有需要的数据
    detection_stores = np.array([face_record[1].det_score for face_record in female_face_records], dtype=np.float64)
    ages = np.array([r[1].age for r in female_face_records], dtype=np.float64)
    embeddings = np.array([r[1].normed_embedding for r in female_face_records], dtype=np.float64)
    poses = np.array([r[1].pose for r in female_face_records], dtype=np.float64)

    # 计算置信度
    detection_confidences = _normalize_min_max(detection_stores)

    pose_confidences = np.abs(poses)
    pose_confidences[:, 2] = pose_confidences[:, 2] * 0.5
    pose_confidences = np.prod(pose_confidences, axis=1) ** (1.0 / 3.0)
    pose_confidences = _normalize_min_max(pose_confidences)

    final_confidences = detection_confidences - pose_confidences

    # 利用人脸向量排除特异点
    mean_embedding = np.mean(embeddings, axis=0)
    similarity_scores = _calculate_cos_similarity(embeddings, mean_embedding)
    valid_mask = np.where(similarity_scores > similarity_scores.mean() - similarity_scores.std(), 1, 0)
    embedding_weights = final_confidences * valid_mask

    embedding_weight_sum  = embedding_weights.sum()
    if embedding_weight_sum <= 0:
        raise FaceNotFoundError()
    # 计算平均年龄
    average_age = (embedding_weights * ages).sum() / embedding_weight_sum

    # 计算人脸向量
    weighted_mean_embedding = (embeddings.T * embedding_weights).sum(axis=1) / embedding_weight_sum

    # 获取最佳人脸
    sorted_indices = np.argsort(final_confidences)
    best_face_index = int(sorted_indices[-1])
    best_frame_index, best_face_record = female_face_records[best_face_index]

    # 返回结果
    best_face = DetectedFace.from_insightface_record(best_face_record)
    if np.isnan(average_age):
        raise FaceNotFoundError()
    res = FaceDetectionResult(best=best_face, frame_index=best_frame_index, age=average_age,
                              face_serial=weighted_mean_embedding)
    return res


def crop_and_rotate_face_into_square(image: np.ndarray, face: DetectedFace) -> np.ndarray:
    face_center_x_y = ((face.left + face.right) // 2, (face.top + face.bottom) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(face_center_x_y, face.roll / 3 * 2, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # 脸部高度占画幅一半,且居中
    new_y1 = int(max(face.top - face.height // 2, 0))
    new_y2 = int(min(new_y1 + face.height * 2, rotated_image.shape[0]))

    # 扩张画幅到1比1
    new_x1 = int(face_center_x_y[0] - (new_y2 - new_y1) / 2)
    new_x2 = new_x1 + (new_y2 - new_y1)

    new_img = rotated_image[new_y1: new_y2, new_x1:new_x2]
    return new_img


def _test_crop():
    from src.file_index import IMAGE_FILE_FOR_TEST
    img = cv2.imread(IMAGE_FILE_FOR_TEST.absolute().__str__())

    face = DetectedFace.from_insightface_record(_FACE_ANALYSER.get(img)[0])

    crop_image = crop_and_rotate_face_into_square(img, face)

    cv2.imshow('', crop_image)
    cv2.waitKey(0)


def _test_detect():
    from src.loader import iter_keyframe_bgr24
    # keyframe_records = list(iter_keyframe_bgr24(VIDEO_FILE_FOR_TEST))
    keyframe_records = list(iter_keyframe_bgr24(pathlib.Path(r'E:\L3\FC2 PPV 1422052 ※期間限定【個人】要介護の旦那を持つ爆乳妻が車内で他人棒に中出しされる.mp4')))
    start_at = time.time()
    out = select_best_female_face([f.bgr_array for f in keyframe_records])
    cost = time.time() - start_at
    print(f'frame_count: {len(keyframe_records)}, cost: {cost:.2f}, mean:{cost / len(keyframe_records):.2f}')
    cv2.imshow('', keyframe_records[out.frame_index].bgr_array)
    cv2.waitKey(0)


if __name__ == '__main__':
    _test_detect()
