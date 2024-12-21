import av
import numpy as np


def get_audio_samples_as_float32_array(file_path, sample_rate=16000, mono=True) -> np.ndarray:
    with av.open(file_path) as container:
        audio_stream = [stream for stream in container.streams if stream.type == 'audio'][0]

        resampler = av.AudioResampler(format='s16', layout='mono' if mono else audio_stream.layout.name,
                                      rate=sample_rate)

        samples = []
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                # Resample the frame to the desired sample rate and format
                frame = resampler.resample(frame)
                # Convert the frame to a NumPy array
                array = frame[0].to_ndarray()
                samples.append(array)

    samples_np = np.hstack(samples).astype(np.float32)
    return samples_np
