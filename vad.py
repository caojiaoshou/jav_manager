import av
import numpy as np
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps
torchaudio.load
model = load_silero_vad()

p = 'E:\L6\FC2-PPV-3089570\hhd800.com@FC2-PPV-3089570.mp4'


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
wav /= np.iinfo(np.int16).max
wav = torch.from_numpy(wav)
wav = wav.unsqueeze(0)
# wav = read_audio(p)

for i in [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=i,
        return_seconds=False,
        visualize_probs=False# Return speech timestamps in seconds (default is samples)
    )
    print(i)
    print(speech_timestamps.__len__())

