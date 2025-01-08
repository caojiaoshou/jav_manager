import contextlib
import datetime
import enum
import typing as t

from sqlmodel import SQLModel, Field, create_engine, Session

from src.file_index import DATABASE_STORAGE


class _Progress(enum.IntEnum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2


class _Videos(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    dir_path: str = Field(index=True)
    file_name: str = Field(index=True)
    vad_state: _Progress = Field(default=_Progress.NOT_STARTED)
    asr_state: _Progress = Field(default=_Progress.NOT_STARTED)
    translate_state: _Progress = Field(default=_Progress.NOT_STARTED)
    preview_state: _Progress = Field(default=_Progress.NOT_STARTED)
    sticker_state: _Progress = Field(default=_Progress.NOT_STARTED)
    scene_state: _Progress = Field(default=_Progress.NOT_STARTED)


class _QuickLooks(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(default=None, index=True)
    preview_path: str
    sticker_path: str


class _Speeches(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(default=None, index=True)
    start_at: float
    end_at: float
    asr_text: str
    translate_text: str


class _Scenes(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(default=None, index=True)
    start_at: float
    preview_path: str


class _AudioSamplePickles(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(default=None, index=True)
    path: str


class _SlaveHistory(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(default=None, index=True)
    acquire_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    release_at: datetime.datetime = Field(default=datetime.datetime.min)
    processor: str = Field(default='')
    result: str = Field(default='')


_DB_PATH = DATABASE_STORAGE / 'database.db'
_ENGINE = create_engine(f'sqlite:///{_DB_PATH.absolute().__str__()}')
SQLModel.metadata.create_all(_ENGINE)


@contextlib.contextmanager
def _use_session() -> t.ContextManager[Session]:
    with Session(_ENGINE) as session:
        try:
            yield session
        except Exception as e:
            raise e
        finally:
            session.close()
