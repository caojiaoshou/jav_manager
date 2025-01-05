import typing as t


class Middleware(t.NamedTuple):
    start_at: float
    end_at: float
    transcribe_text: str
    translate_text: str
