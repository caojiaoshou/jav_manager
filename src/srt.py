import pickle
from datetime import datetime, timezone

with open('../translation.pickle', 'rb') as f:
    result = pickle.load(f)


def format_time(time: float) -> str:
    dt = datetime.fromtimestamp(time, timezone.utc)
    return dt.strftime('%H:%M:%S,%f')


lines = []
for segment in result['segments']:
    lines.append(str(segment['id']))

    start = format_time(segment['start'])

    end = format_time(segment['end'])

    time_line = f'{start} --> {end}'
    lines.append(time_line)

    lines.append(segment['translation'])

    lines.append('\n')

with open('../fuck.srt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
