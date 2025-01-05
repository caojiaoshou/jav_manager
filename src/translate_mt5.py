import gc
import typing

import torch
from transformers import pipeline, Pipeline

from src.file_index import MODEL_STORAGE

_DOWNLOAD_DIR = (MODEL_STORAGE / 'transformer').absolute().__str__()


def pip_factory() -> Pipeline:
    """
    由于硬件性能限制,模型不应该作为模块变量,这会极大的浪费内存,造成其它模块运行缓慢,只能牺牲IO
    :return:
    """
    return pipeline(model="larryvrh/mt5-translation-ja_zh", model_kwargs={'cache_dir': _DOWNLOAD_DIR})


def translate_list(list_to_translate: typing.Iterable[str]) -> list[str]:
    """
    :param list_to_translate: 模型好像限制单词输入限制输入约为120token
    :return:
    """
    pad_request = [f'<-ja2zh-> {s}' for s in list_to_translate]
    pipe = pip_factory()
    result_list = []
    # 不要用dataset的方式.数据量上去后实测会卡死!! 15.1GB VRAM + 94%GPU 已然冒烟
    for text in pad_request:
        text = ''.join(pipe.tokenizer.tokenize(text)[:int(128 * 0.9)])
        result = pipe(text)
        print(result)
        result_list.extend(result)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return [r['translation_text'] for r in result_list]


if __name__ == '__main__':
    test_ls = [
        '心が動くことを知らないで、あなたの美しい眉の目だけを見て、万水千山を見たことがあるようです。',
        '悲しまないでください。凧は風があります。イルカは海があります。あなたも私がいます。',
        '私を選んだことで幸せを感じることができます。'
    ]
    print(translate_list(test_ls))
