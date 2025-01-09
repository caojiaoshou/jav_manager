from pathlib import Path
from typing import Final

_ROOT_DIR = Path(__file__).parents[1]

MODEL_STORAGE: Final[Path] = _ROOT_DIR / 'model'
MODEL_STORAGE.mkdir(exist_ok=True)

TEMP_STORAGE: Final[Path] = _ROOT_DIR / 'sample'
TEMP_STORAGE.mkdir(exist_ok=True)

GUI_STORAGE: Final[Path] = _ROOT_DIR / 'gui_dist'
GUI_STORAGE.mkdir(exist_ok=True)

DATABASE_STORAGE = _ROOT_DIR / 'database'
DATABASE_STORAGE.mkdir(exist_ok=True)

LOG_STORAGE: Final[Path] = _ROOT_DIR / 'log'
LOG_STORAGE.mkdir(exist_ok=True)

VIDEO_DIR_FOR_TEST: Final[Path] = Path(r'E:\L6\[98t.tv]FC2PPV-3009465')
VIDEO_FILE_FOR_TEST: Final[Path] = VIDEO_DIR_FOR_TEST / 'FC2PPV-3009465-1.mp4'
IMAGE_FILE_FOR_TEST: Final[Path] = Path(r'C:\Users\Administrator\Desktop\GXM0AXwacAENNuM.jpg')

TOP_DIR_LIST: Final[list[Path]] = [
    Path(r'E:\L3'),
    Path(r'E:\L6'),
    Path(r'D:\L'),
    Path(r'D:\L3.5'),
    Path(r'D:\L5'),
]


def search_local_videos() -> list[tuple[str, Path]]:
    res = []
    for rd in TOP_DIR_LIST:
        res.extend(_search_videos_in_single_root(rd))
    return res


def _search_videos_in_single_root(root_dir: Path) -> list[tuple[str, Path]]:
    res = []
    target_suffix = ['.mp4', '.avi', '.mkv']
    for f_o_d in root_dir.iterdir():
        if f_o_d.is_dir():
            for f_inner in f_o_d.iterdir():
                if f_inner.is_file() and f_inner.suffix in target_suffix:
                    res.append((f_o_d.name, f_inner))
        elif f_o_d.is_file() and f_o_d.suffix in target_suffix:
            res.append((f_o_d.with_suffix('').name, f_o_d))
        else:
            continue
    return res


if __name__ == '__main__':
    print(search_local_videos())
