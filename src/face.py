import pathlib

import cv2
from insightface.app import FaceAnalysis

_MODEL_DIR = pathlib.Path(__file__).parents[1] / 'model' / 'insightface'
_MODEL_DIR.mkdir(exist_ok=True)
_ANALYSER = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                         root=_MODEL_DIR.absolute().__str__())
_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))


def detect():
    img = cv2.imread(r'C:\Users\Administrator\Desktop\GXM0AXwacAENNuM.jpg')
    faces = _ANALYSER.get(img)
    """
    Pitch (Up/Down), Yaw (Left/Right),Roll (Tilt)
    """
    pitch, yar, roll = faces[0].pose

    """
    left_eye, right_eye, nose, left_mouth, right_mouth
    """
    kps = faces[0].kps
    print(faces)

    faces[0].normed_embedding
    rimg = _ANALYSER.draw_on(img, faces)
    cv2.imshow("t1_output.jpg", rimg)
    cv2.waitKey(0)


if __name__ == '__main__':
    detect()
