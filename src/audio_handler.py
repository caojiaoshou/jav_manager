import pathlib

from src.asr import create_transcribe
from src.config import Middleware
from src.loader import get_audio_samples_as_float32_array
from src.srt import create_srt_content
from src.translate_mt5 import translate_list
from src.vad import create_vad


def full_work(p_todo: pathlib.Path):
    # 文件IO
    audio_array = get_audio_samples_as_float32_array(p_todo)

    # 使用vad启发有效片段
    vad = create_vad(audio_array)

    # asr识别
    transcribe_result = create_transcribe(audio_array, vad)

    # 翻译
    to_trans = [transcribe_item.transcribe_text for transcribe_item in transcribe_result]
    translate_result = translate_list(to_trans)
    map_list = [Middleware(*raw[0:3], trans) for raw, trans in zip(transcribe_result, translate_result)]

    # 输出字幕
    srt_content = create_srt_content(map_list)

    srt_path = pathlib.Path(p_todo).with_suffix('.srt')
    srt_path.write_text(srt_content, encoding='utf-8')


if __name__ == '__main__':
    ls = pathlib.Path(r'E:\L6\FC2-PPV-3939370').glob('*.mp4')
    for p in ls:
        full_work(p)
