import gc

import numpy as np
import torch
import whisper

from src.config import Middleware, MODEL_DIR


def create_transcribe(wav_array: np.ndarray, heuristic_cut: list[Middleware] | None = None) -> list[Middleware]:
    """
    v3对于非英语效果更好,显存需求10G
    v3turbo对非英语效果有一定下降,显存需求6G,且拥有8x推理速度
    :param wav_array:
    :param heuristic_cut:
    :return:
    """
    asr_model = whisper.load_model('large-v3', download_root=MODEL_DIR.absolute().__str__())

    transcript_list = []
    for speech_timestamp in heuristic_cut:
        start_at = speech_timestamp.start_at - 0.1
        end_at = speech_timestamp.end_at + 0.1

        # 除32786参照whisper的load_audio来的.不知道是为什么
        audio = wav_array[:, int(start_at * 16000): int(end_at * 16000)].flatten() / 32786
        result = asr_model.transcribe(audio, language='ja')
        print(result)
        if transcribe_text := result['text']:
            transcript_list.append(Middleware(start_at, end_at, transcribe_text, ''))
    # 确保释放显存
    del asr_model
    gc.collect()
    torch.cuda.empty_cache()
    return transcript_list
