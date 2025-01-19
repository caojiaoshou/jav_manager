import functools
import time

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from src.config import Middleware
from src.logger import configure_logger

_logger = configure_logger('audio')


@functools.lru_cache(1)
def vad_factory():
    _model_loading_start_at = time.time()
    _vad_model = load_silero_vad()
    _logger.debug(f'silero_vad model loading time: {time.time() - _model_loading_start_at:0.2f}s')
    return _vad_model


def create_vad(sample_array: np.ndarray) -> list[Middleware]:
    silero_style_sample = sample_array / np.iinfo(np.int16).max
    silero_style_sample = torch.from_numpy(silero_style_sample)
    speech_timestamps = get_speech_timestamps(
        silero_style_sample,
        vad_factory(),
        threshold=0.35,  # 默认是0.5,实测下来在有底噪的情况下(AV通病),会有不少假阴. 0.3的话好像又有不少假阳.折中一下
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )

    return [Middleware(d['start'], d['end'], '', '') for d in speech_timestamps]
