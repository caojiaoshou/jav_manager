import datetime
import pathlib
import pickle
from pprint import pprint

import torch
import whisper
from src import MODEL_DIR
print(torch.cuda.is_available())

p = pathlib.Path(r'E:\L6\FC2-PPV-3089570\hhd800.com@FC2-PPV-3089570.mp4')
model = whisper.load_model('turbo', download_root=MODEL_DIR.absolute().__str__())

load_start_at = datetime.datetime.now()
print(f"{load_start_at=}")
audio = whisper.load_audio(p)
load_finish_at = datetime.datetime.now()
print(f"{load_finish_at=}")

transcript_start_at = datetime.datetime.now()
print(f"{transcript_start_at=}")
result = model.transcribe(audio, language='ja')
transcript_finish_at = datetime.datetime.now()
print(f"{transcript_finish_at=}")
print((transcript_start_at - transcript_finish_at).total_seconds())

with open('../result.pickle', 'wb') as io:
    pickle.dump(result, io)
pprint(result)
