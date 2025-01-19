import gc
import time

import numpy as np
import torch
import whisper

from src.config import Middleware
from src.file_index import MODEL_STORAGE
from src.logger import configure_logger
from src.vad import create_vad

_logger = configure_logger('audio')


def create_transcribe(wav_array: np.ndarray, heuristic_cut: list[Middleware] | None = None) -> list[Middleware]:
    """
    v3对于非英语效果更好,显存需求10G
    v3turbo对非英语效果有一定下降,显存需求6G,且拥有8x推理速度
    :param wav_array:
    :param heuristic_cut:
    :return:
    """
    # 添加非引导的情况
    if heuristic_cut is None:
        heuristic_cut = create_vad(wav_array)

    # 对于AV.一些空的情况就不处理了
    if not heuristic_cut:
        _logger.debug('vad empty ')
        return []

    load_start_at = time.time()
    asr_model = whisper.load_model('large-v3', download_root=MODEL_STORAGE.absolute().__str__())
    _logger.debug(f'asr model loading time: {time.time() - load_start_at:0.2f}s')

    transcript_list = []
    asr_start_at = time.time()
    asr_size = 0
    for speech_timestamp in heuristic_cut:
        start_at = speech_timestamp.start_at - 0.1
        end_at = speech_timestamp.end_at + 0.1
        asr_size += end_at - start_at
        # 除32786参照whisper的load_audio来的.不知道是为什么
        # 牛逼千问直接发现了这个typo的bug. 32768打成32786
        audio = wav_array[:, int(start_at * 16000): int(end_at * 16000)].flatten() / 32768.0
        result = asr_model.transcribe(audio, language='ja')
        if transcribe_text := result['text']:
            transcript_list.append(Middleware(start_at, end_at, transcribe_text, ''))
    asr_cost = time.time() - asr_start_at
    _logger.debug(f'asr cost: {asr_cost:.2f}, asr size: {asr_size:.2f}, mean cost: {asr_cost / asr_size:.2f}')
    # 确保释放显存
    del asr_model
    gc.collect()
    torch.cuda.empty_cache()
    return transcript_list
