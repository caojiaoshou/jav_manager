import contextlib
import datetime
import enum
import pathlib
import typing as t

import numpy as np
from pydantic import ConfigDict
from sqlalchemy import TypeDecorator, String, Column, BLOB
from sqlmodel import SQLModel, Field, create_engine, Session

from src.file_index import DATABASE_STORAGE


class PathType(TypeDecorator):
    impl = String

    def process_bind_param(self, value, dialect):
        if isinstance(value, pathlib.Path):
            return value.absolute().__str__()
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return pathlib.Path(value)
        return value


class Float32ArrayType(TypeDecorator):
    impl = BLOB

    def process_bind_param(self, value, dialect):
        if isinstance(value, np.ndarray):
            return value.astype(np.float32).tobytes()
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return np.frombuffer(value, dtype=np.float32)
        return value


class _Progress(enum.IntEnum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2


class _Videos(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    group: str = Field(index=True)
    file_path: pathlib.Path = Field(sa_column=Column(PathType))
    vad_state: int = Field(default=_Progress.NOT_STARTED)
    asr_state: int = Field(default=_Progress.NOT_STARTED)
    translate_state: int = Field(default=_Progress.NOT_STARTED)
    quick_look_state: int = Field(default=_Progress.NOT_STARTED)
    face_state: int = Field(default=_Progress.NOT_STARTED)
    scene_state: int = Field(default=_Progress.NOT_STARTED)
    delete: bool = Field(default=False)


class _Faces(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(default=None, index=True)
    embedding: np.ndarray = Field(sa_column=Column(Float32ArrayType))
    age: float
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


class _QuickLooks(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    path: pathlib.Path


class _Speeches(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    start_at: float
    end_at: float
    asr_text: str
    translate_text: str


class _Scenes(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    start_at: float
    preview_path: str


class _AudioSamplePickles(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    path: pathlib.Path


class _SlaveHistory(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
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


def add_video(group: str, file_path: pathlib.Path):
    with _use_session() as session:
        session.add(_Videos(group=group, file_path=file_path))
        session.commit()


def delete_video(video_id: int):
    with _use_session() as session:
        session.query(_Videos).filter(_Videos.id == video_id).update({_Videos.delete: True})
        session.commit()


def list_videos() -> list[_Videos]:
    with _use_session() as session:
        return session.query(_Videos).filter(_Videos.delete == False).all()
