import datetime
import hashlib
import re
import time

import src.dao as dao
from src.audio_handler import audio_full_work
from src.file_index import search_local_videos
from src.loader import get_video_duration
from src.logger import configure_logger
from src.video_handler import video_full_work

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


def prepare_videos():
    current_local_mapper = {local_path.absolute(): dir_name for dir_name, local_path in search_local_videos()}
    _LOGGER.info(f'本地共 {len(current_local_mapper)} 个视频')

    current_db_mapper = {r.file_path: r.id for r in dao.list_videos()}
    _LOGGER.info(f'数据库共 {len(current_db_mapper)} 个视频')

    for local_path, dir_name in current_local_mapper.items():
        if local_path not in current_db_mapper:
            video_duration = get_video_duration(local_path)
            is_video_valid = video_duration >= 60 * 4

            if is_video_valid:
                video_hash = hashlib.md5(local_path.read_bytes()).hexdigest()
                dao.add_video(
                    _sanitize_name(dir_name),
                    local_path,
                    datetime.datetime.fromtimestamp(local_path.stat().st_ctime),
                    video_duration,
                    video_hash,
                    local_path.stat().st_size
                )

                _LOGGER.info(f'往数据库添加了 {local_path}')
            else:
                _LOGGER.warning(f'{local_path} 不符合标准，跳过')

    for local_path, id_ in current_db_mapper.items():
        if local_path not in current_local_mapper:
            dao.delete_video(id_)
            _LOGGER.info(f'从数据库删除了 {local_path}')


def handle_views():
    prepare_videos()
    db_videos = sorted(dao.list_videos(), key=lambda x: x.file_create_at, reverse=True)
    _LOGGER.info(f'开始生成预览. 共 {len(db_videos)} 个视频')
    for r in db_videos:
        video_progress_state_value = dao.calculate_video_progress_state(r)
        match video_progress_state_value:
            case dao.ProgressState.NOT_STARTED:
                _LOGGER.info(f'开始处理 {r.file_path}')
                start_at = time.time()
                full_work_result = video_full_work(r.file_path)
                _LOGGER.info(f'完成处理 {r.file_path} 用时 {time.time() - start_at:.2f}s')
                dao.update_quick_look(r.id, full_work_result.quick_look)
                dao.update_body_part(r.id, full_work_result.body_parts)
                dao.update_scene(r.id, full_work_result.scenes)
                dao.update_face(r.id, full_work_result.faces)
                _LOGGER.info(f'往数据库写入 {r.file_path} 的视频信息')
            case dao.ProgressState.IN_PROGRESS:
                _LOGGER.warning(f'开始删除 {r.file_path} 的视频信息.可能是由于此前错误退出')
                start_at = time.time()
                dao.delete_quick_look(r.id)
                dao.delete_body_part(r.id)
                dao.delete_scene(r.id)
                dao.delete_face(r.id)
                _LOGGER.warning(f'完成删除 {r.file_path} 的视频信息 用时 {time.time() - start_at:.2f}s')
            case dao.ProgressState.COMPLETED:
                _LOGGER.info(f'忽略 {r.file_path} 此前处理完成')
                continue
            case _:
                _LOGGER.error(f'未知的进度状态 {video_progress_state_value}')
                raise ValueError(f'unknown progress state {video_progress_state_value}')


def handle_srt():
    for video_ist in sorted(dao.list_videos(), key=lambda x: x.file_create_at, reverse=True):
        video_state = dao.calculate_video_progress_state(video_ist)
        if video_state != dao.ProgressState.COMPLETED:
            continue

        audio_state = dao.calculate_audio_progress_state(video_ist)
        match audio_state:
            case dao.ProgressState.NOT_STARTED:
                _LOGGER.info(f'开始处理 {video_ist.file_path} 的字幕')
                start_at = time.time()
                middle_ware_ls = audio_full_work(video_ist.file_path)
                dao.update_srt(video_ist.id, middle_ware_ls)
                _LOGGER.info(f'完成处理 {video_ist.file_path} 的字幕 用时 {time.time() - start_at:.2f}s')
            case dao.ProgressState.IN_PROGRESS:
                _LOGGER.warning(f'开始删除 {video_ist.file_path} 的字幕.可能是由于此前错误退出')
                start_at = time.time()
                dao.delete_srt(video_ist.id)
                _LOGGER.warning(f'完成删除 {video_ist.file_path} 的字幕 用时 {time.time() - start_at:.2f}s')
            case dao.ProgressState.COMPLETED:
                _LOGGER.info(f'忽略 {video_ist.file_path} 此前处理完成')
                continue
            case _:
                _LOGGER.error(f'未知的进度状态 {audio_state}')
                raise ValueError(f'unknown progress state {audio_state}')


def list_finish_views() -> list[id]:
    return [r.id for r in dao.list_videos() if dao.calculate_video_progress_state(r) == dao.ProgressState.COMPLETED]


if __name__ == '__main__':
    handle_srt()
    # handle_views()
