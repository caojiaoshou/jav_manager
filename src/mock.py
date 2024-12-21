from src.asr import create_transcribe
from src.loader import get_audio_samples_as_float32_array
from src.srt import create_srt_content
from src.translate_mt5 import translate_list
from src.vad import create_vad

p_for_test = 'D:\L5\[JAV] [Uncensored] FC2 PPV 1888207 [1080p]\FC2-PPV-1888207_1.mp4'

# 文件IO
audio_array = get_audio_samples_as_float32_array(p_for_test)

# 使用vad启发有效片段
vad = create_vad(audio_array)

# asr识别
transcribe = create_transcribe(audio_array, vad)

# 翻译
to_trans = [transcribe_item.transcribe_text for transcribe_item in transcribe]
map_list = [(*raw[0:3], trans) for raw, trans in zip(transcribe, translate_list(to_trans))]

# 输出字幕
srt_content = create_srt_content(map_list)
print(srt_content)
