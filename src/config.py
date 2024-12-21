import pathlib
import typing
import typing as t

MODEL_DIR: typing.Final[pathlib.Path] = pathlib.Path(__file__).parents[1] / 'model'


class Middleware(t.NamedTuple):
    start_at: float
    end_at: float
    transcribe_text: str
    translate_text: str
