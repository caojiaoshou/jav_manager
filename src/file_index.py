from pathlib import Path
from typing import Final

_ROOT_DIR = Path(__file__).parents[1]

MODEL_STORAGE: Final[Path] = _ROOT_DIR / 'model'
MODEL_STORAGE.mkdir(exist_ok=True)

TEMP_STORAGE: Final[Path] = _ROOT_DIR / 'sample'
TEMP_STORAGE.mkdir(exist_ok=True)

GUI_STORAGE: Final[Path] = _ROOT_DIR / 'gui_dist'
GUI_STORAGE.mkdir(exist_ok=True)

VIDEO_DIR_FOR_TEST: Final[Path] = Path(r'E:\L6\[98t.tv]FC2PPV-3009465')
VIDEO_FILE_FOR_TEST: Final[Path] = VIDEO_DIR_FOR_TEST / 'FC2PPV-3009465-1.mp4'
IMAGE_FILE_FOR_TEST: Final[Path] = Path(r'C:\Users\Administrator\Desktop\GXM0AXwacAENNuM.jpg')
