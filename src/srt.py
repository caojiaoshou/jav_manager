import datetime
import typing as t

from src.config import Middleware


def _format_time(time_: float) -> str:
    dt = datetime.datetime.fromtimestamp(time_, datetime.timezone.utc)
    return dt.strftime('%H:%M:%S,%f')


def create_srt_content(middleware_seq: t.Sequence[Middleware]) -> str:
    lines = []
    for id_, (start, end, original_text, translate_text) in enumerate(middleware_seq, start=1):
        lines.append(str(id_))

        start = _format_time(start)

        end = _format_time(end)

        time_line = f'{start} --> {end}'
        lines.append(time_line)

        lines.append(translate_text)
        lines.append(original_text)

        lines.append('\n')

    return '\n'.join(lines)
