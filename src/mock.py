import datetime
import pathlib

import av
import numpy as np
import torch
import whisper
from silero_vad import load_silero_vad, get_speech_timestamps

from translate_v2 import translate_sentence

p = 'D:\L5\[JAV] [Uncensored] FC2 PPV 1888207 [1080p]\FC2-PPV-1888207_1.mp4'
vad_model = load_silero_vad(onnx=True)


def get_audio_samples(file_path, sample_rate=16000, mono=True):
    container = av.open(file_path)
    audio_stream = next(stream for stream in container.streams if stream.type == 'audio')

    resampler = av.AudioResampler(format='s16', layout='mono' if mono else audio_stream.layout.name, rate=sample_rate)

    samples = []
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            # Resample the frame to the desired sample rate and format
            frame = resampler.resample(frame)
            # Convert the frame to a NumPy array
            array = frame[0].to_ndarray()
            samples.append(array)

    samples_np = np.hstack(samples)
    return samples_np


wav = get_audio_samples(p)
wav = wav.astype(np.float32)
wav_2 = wav / np.iinfo(np.int16).max
wav_2 = torch.from_numpy(wav_2)
# wav = wav.unsqueeze(0)
# wav = read_audio(p)
speech_timestamps = get_speech_timestamps(
    wav_2,
    vad_model,
    threshold=0.3,
    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)
print(speech_timestamps)

print(torch.cuda.is_available())

p = pathlib.Path(r'E:\L6\FC2-PPV-3089570\hhd800.com@FC2-PPV-3089570.mp4')
asr_model = whisper.load_model('large-v3', download_root='model/')

transcript_list = []
for speech_timestamp in speech_timestamps:
    start_at = speech_timestamp['start'] - 0.1
    end_at = speech_timestamp['end'] + 0.1
    audio = wav[:, int(start_at * 16000): int(end_at * 16000)].flatten() / 32786
    transcript_start_at = datetime.datetime.now()
    result = asr_model.transcribe(audio, language='ja')
    print(result)
    transcript_list.append((start_at, end_at, result['text']))

map_list = []
# for (start, end, original_text) in transcript_list:
#     while True:
#         # text = segment['text']
#         salt = uuid4().hex
#         appid = '20241218002231705'
#         key = 'QMXTG9JvmiLklqGUu223'
#
#         concat_sign = appid + original_text + salt + key
#
#         sign = hashlib.md5(concat_sign.encode('utf-8')).hexdigest()
#         query = {
#             'q': original_text,
#             'from': 'jp',
#             'to': 'zh',
#             'appid': appid,
#             'salt': salt,
#             'sign': sign
#         }
#         result_2 = httpx.get('https://fanyi-api.baidu.com/api/trans/vip/translate', params=query)
#         print(result_2.status_code)
#         result_3 = result_2.json()
#         print(result_3)
#         try:
#             translate_text = result_2.json()['trans_result'][0]['dst']
#             map_list.append((start, end, original_text, translate_text))
#             break
#         except KeyError:
#             ...
#         finally:
#             time.sleep(0.2)
for (start, end, original_text) in transcript_list:
    map_list.append((start, end, original_text, translate_sentence(original_text)))


def format_time(time_: float) -> str:
    dt = datetime.datetime.fromtimestamp(time_, datetime.timezone.utc)
    return dt.strftime('%H:%M:%S,%f')


lines = []
for id_, (start, end, original_text, translate_text) in enumerate(map_list):
    lines.append(str(id_))

    start = format_time(start)

    end = format_time(end)

    time_line = f'{start} --> {end}'
    lines.append(time_line)

    lines.append(translate_text)
    lines.append(original_text)

    lines.append('\n')

with open('../fuck.srt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
