import pathlib
import re
import time

from src.dao import list_videos, add_video, delete_video, ProgressState, update_face, update_scene, update_body_part, \
    update_quick_look, delete_quick_look, delete_body_part, delete_face, delete_scene
from src.file_index import search_local_videos
from src.loader import get_video_duration
from src.logger import configure_logger
from src.video_handler import video_full_work, calculate_video_progress_state

_LOGGER = configure_logger('master')


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
    return get_video_duration(p) >= 60 * 4


def prepare_videos():
    current_local_mapper = {local_path.absolute(): dir_name for dir_name, local_path in search_local_videos()}
    _LOGGER.info(f'本地共 {len(current_local_mapper)} 个视频')

    current_db_mapper = {r.file_path: r.id for r in list_videos()}
    _LOGGER.info(f'数据库共 {len(current_db_mapper)} 个视频')

    for local_path, dir_name in current_local_mapper.items():
        if local_path not in current_db_mapper:
            if _is_video_valid(local_path):
                add_video(_sanitize_name(dir_name), local_path)
                _LOGGER.info(f'往数据库添加了 {local_path}')
            else:
                _LOGGER.warning(f'{local_path} 不符合标准，跳过')

    for local_path, id_ in current_db_mapper.items():
        if local_path not in current_local_mapper:
            delete_video(id_)
            _LOGGER.info(f'从数据库删除了 {local_path}')


def handle_views():
    prepare_videos()
    db_videos = list_videos()
    _LOGGER.info(f'开始生成预览. 共 {len(db_videos)} 个视频')
    for r in list_videos():
        video_progress_state_value = calculate_video_progress_state(r)
        match video_progress_state_value:
            case ProgressState.NOT_STARTED:
                _LOGGER.info(f'开始处理 {r.file_path}')
                start_at = time.time()
                full_work_result = video_full_work(r.file_path)
                _LOGGER.info(f'完成处理 {r.file_path} 用时 {time.time() - start_at:.2f}s')
                update_quick_look(r.id, full_work_result.quick_look)
                update_body_part(r.id, full_work_result.body_parts)
                update_scene(r.id, full_work_result.scenes)
                update_face(r.id, full_work_result.faces)
                _LOGGER.info(f'往数据库写入 {r.file_path} 的视频信息')
            case ProgressState.IN_PROGRESS:
                _LOGGER.warning(f'开始删除 {r.file_path} 的视频信息.可能是由于此前错误退出')
                start_at = time.time()
                delete_quick_look(r.id)
                delete_body_part(r.id)
                delete_scene(r.id)
                delete_face(r.id)
                _LOGGER.warning(f'完成删除 {r.file_path} 的视频信息 用时 {time.time() - start_at:.2f}s')
            case ProgressState.COMPLETED:
                _LOGGER.info(f'忽略 {r.file_path} 此前处理完成')
                continue
            case _:
                _LOGGER.error(f'未知的进度状态 {video_progress_state_value}')
                raise ValueError(f'unknown progress state {video_progress_state_value}')


def list_finish_views() -> list[id]:
    return [r.id for r in list_videos() if calculate_video_progress_state(r) == ProgressState.COMPLETED]


if __name__ == '__main__':
    prepare_videos()
