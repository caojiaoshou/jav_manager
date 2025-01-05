import os
from pathlib import Path

import cv2
import numpy as np
from rembg import new_session, remove


def get_foreground_mask(image: np.ndarray) -> np.ndarray:
    model_dir = Path(__file__).parents[1] / 'model' / 'rmbg'
    model_dir.mkdir(exist_ok=True)
    os.environ['U2NET_HOME'] = model_dir.absolute().__str__()
    session = new_session(model_name='birefnet-portrait')
    return remove(image, only_mask=True, session=session)


def _test_rmbg():
    p_for_test = Path(r'C:\Users\Administrator\Desktop\GXM0AXwacAENNuM.jpg')
    result = get_foreground_mask(cv2.imread(p_for_test, cv2.IMREAD_COLOR))
    cv2.imshow('', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    _test_rmbg()
