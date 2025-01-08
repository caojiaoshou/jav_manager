import pathlib
import re

from src.dao import list_videos, add_video, delete_video
from src.file_index import search_local_videos
from src.loader import get_video_duration


def _remove_bracketed_content(input_string: str) -> str:
    # 使用正则表达式匹配成对出现的中括号及其内容
    pattern = r'\[[^\[\]]*\]'
    # 替换匹配到的内容为空字符串
    result = re.sub(pattern, '', input_string)
    return result


def _remove_non_ascii(input_string: str) -> str:
    # 使用正则表达式匹配非ASCII字符
    pattern = r'[^\x00-\x7F]+'
    # 替换匹配到的内容为空字符串
    result = re.sub(pattern, '', input_string)
    return result


def _sanitize_name(name: str) -> str:
    end = _remove_non_ascii(_remove_bracketed_content(name)).strip()
    return end if end else name


def _is_video_valid(p: pathlib.Path) -> bool:
    return get_video_duration(p) >= 60 * 5


def prepare_videos():
    current_local_mapper = {p.absolute(): name for name, p in search_local_videos()}
    current_db_mapper = {r.file_path: r.id for r in list_videos()}

    for p, name in current_local_mapper.items():
        if (p not in current_db_mapper) and (_is_video_valid(p)):
            add_video(_sanitize_name(name), p)

    for p, id_ in current_db_mapper.items():
        if p not in current_local_mapper:
            delete_video(id_)


if __name__ == '__main__':
    prepare_videos()
