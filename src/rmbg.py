import os

import cv2
import numpy as np
from rembg import new_session, remove

from src.file_index import MODEL_STORAGE, IMAGE_FILE_FOR_TEST


def get_foreground_mask(image: np.ndarray) -> np.ndarray:
    model_dir = MODEL_STORAGE / 'rmbg'
    model_dir.mkdir(exist_ok=True)
    os.environ['U2NET_HOME'] = model_dir.absolute().__str__()
    session = new_session(model_name='birefnet-portrait', providers=['CUDAExecutionProvider'])
    return remove(image, only_mask=True, session=session)


def _test_rmbg():
    result = get_foreground_mask(cv2.imread(IMAGE_FILE_FOR_TEST, cv2.IMREAD_COLOR))
    cv2.imshow('', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    _test_rmbg()
