import pathlib
import time

from src.asr import create_transcribe
from src.config import Middleware
from src.file_index import VIDEO_DIR_FOR_TEST
from src.loader import get_audio_samples_as_float32_array
from src.logger import configure_logger
from src.srt import create_srt_content
from src.translate_mt5 import translate_list
from src.vad import create_vad

_logger = configure_logger('audio')


def audio_full_work(p_todo: pathlib.Path) -> list[Middleware]:
    # 文件IO
    io_start_at = time.time()
    audio_array = get_audio_samples_as_float32_array(p_todo)
    _logger.debug(f'io cost {time.time() - io_start_at:.2f}s')

    # 使用vad启发有效片段
    vad_start_at = time.time()
    vad = create_vad(audio_array)
    _logger.debug(f'vad cost {time.time() - vad_start_at:.2f}s')

    # asr识别
    asr_start_at = time.time()
    transcribe_result = create_transcribe(audio_array, vad)
    _logger.debug(f'asr cost {time.time() - asr_start_at:.2f}s')

    # 翻译
    translate_start_at = time.time()
    to_trans = [transcribe_item.transcribe_text for transcribe_item in transcribe_result]
    translate_result = translate_list(to_trans)
    map_list = [Middleware(*raw[0:3], trans) for raw, trans in zip(transcribe_result, translate_result)]
    _logger.debug(f'translate cost {time.time() - translate_start_at:.2f}s')
    return map_list


def _test():
    ls = VIDEO_DIR_FOR_TEST.glob('*.mp4')
    for p in ls:
        middle_list = audio_full_work(p)
        # 输出字幕
        srt_content = create_srt_content(middle_list)
        srt_path = pathlib.Path(p).with_suffix('.srt')
        srt_path.write_text(srt_content, encoding='utf-8')


if __name__ == '__main__':
    _test()
