import pathlib
import re

from src.dao import list_videos, add_video, delete_video, ProgressState, update_face, update_scene, update_body_part, \
    update_quick_look, delete_quick_look, delete_body_part, delete_face, delete_scene
from src.file_index import search_local_videos
from src.loader import get_video_duration
from src.video_handler import video_full_work, calculate_video_progress_state


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


def handle_views():
    for r in list_videos():
        video_progress_state_value = calculate_video_progress_state(r)
        match video_progress_state_value:
            case ProgressState.NOT_STARTED:
                full_work_result = video_full_work(r.file_path)
                update_quick_look(r.id, full_work_result.quick_look)
                update_body_part(r.id, full_work_result.body_parts)
                update_scene(r.id, full_work_result.scenes)
                update_face(r.id, full_work_result.faces)
            case ProgressState.IN_PROGRESS:
                delete_quick_look(r.id)
                delete_body_part(r.id)
                delete_scene(r.id)
                delete_face(r.id)
            case ProgressState.COMPLETED:
                continue
            case _:
                raise ValueError(f'unknown progress state {video_progress_state_value}')


def list_finish_views() -> list[id]:
    return [r.id for r in list_videos() if calculate_video_progress_state(r) == ProgressState.COMPLETED]


if __name__ == '__main__':
    handle_views()
