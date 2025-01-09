import contextlib
import datetime
import enum
import pathlib
import threading
import typing as t
import uuid

import cv2
import numpy as np
from pydantic import ConfigDict
from sqlalchemy import TypeDecorator, String, Column, BLOB
from sqlmodel import SQLModel, Field, create_engine, Session, select

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


class ProgressState(enum.IntEnum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2


class Videos(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    group: str = Field(index=True)
    file_path: pathlib.Path = Field(sa_column=Column(PathType))
    vad_state: int = Field(default=ProgressState.NOT_STARTED)
    asr_state: int = Field(default=ProgressState.NOT_STARTED)
    translate_state: int = Field(default=ProgressState.NOT_STARTED)
    quick_look_state: int = Field(default=ProgressState.NOT_STARTED)
    face_state: int = Field(default=ProgressState.NOT_STARTED)
    scene_state: int = Field(default=ProgressState.NOT_STARTED)
    body_part_state: int = Field(default=ProgressState.NOT_STARTED)
    delete: bool = Field(default=False)


class _Faces(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    embedding: np.ndarray = Field(sa_column=Column(Float32ArrayType))
    preview_path: pathlib.Path = Field(sa_column=Column(PathType))
    sticker_path: pathlib.Path = Field(sa_column=Column(PathType))
    age: float
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


class _QuickLooks(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    path: pathlib.Path


class _Speeches(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    start_at: float
    end_at: float
    asr_text: str
    translate_text: str


class _Scenes(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    start_at: float
    preview_path: pathlib.Path = Field(sa_column=Column(PathType))


class _AudioSamplePickles(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    path: pathlib.Path


class _BodyParts(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    part: str
    start_at: float
    frame_path: pathlib.Path = Field(sa_column=Column(PathType))


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

# 储存大头贴和预览视频的地方
_MAJOR_STORAGE = DATABASE_STORAGE / 'major'
_MAJOR_STORAGE.mkdir(exist_ok=True)

# 储存非重要预览的地方
_MINOR_STORAGE = DATABASE_STORAGE / 'minor'
_MINOR_STORAGE.mkdir(exist_ok=True)

_SQLITE_LOCK: threading.RLock = threading.RLock()


class RacingProgressError(Exception):
    ...


@contextlib.contextmanager
def _use_session() -> t.ContextManager[Session]:
    with _SQLITE_LOCK:
        session = Session(_ENGINE)
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


def add_video(group: str, file_path: pathlib.Path):
    with _use_session() as session:
        session.add(Videos(group=group, file_path=file_path))
        session.commit()


def delete_video(video_id: int):
    with _use_session() as session:
        ist = session.exec(select(Videos).filter(Videos.id == video_id)).one()
        ist.delete = True
        session.commit()


def list_videos() -> list[Videos]:
    with _use_session() as session:
        rt = session.exec(select(Videos).filter(Videos.delete == False)).all()
    return rt


class VideoFace(t.NamedTuple):
    embedding: np.ndarray
    age: float
    frame: np.ndarray
    crop_image: np.ndarray


def update_face(video_id: int, face_sequence: t.Iterable[VideoFace]):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        if video_ist.face_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('face block')
        else:
            video_ist.face_state = ProgressState.IN_PROGRESS
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            history_ist = _SlaveHistory(video_id=video_id, processor='face')
            sas.add(history_ist)
            sas.commit()
            for face_record in face_sequence:
                sticker_path = _MAJOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(sticker_path.absolute().__str__(), face_record.crop_image)
                raw_path = _MINOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(raw_path.absolute().__str__(), face_record.frame)
                face_ist = _Faces(
                    video_id=video_id,
                    history_id=history_ist.id,
                    embedding=face_record.embedding,
                    preview_path=raw_path,
                    sticker_path=sticker_path,
                    age=face_record.age
                )
                sas.add(face_ist)
                sas.commit()
        video_ist.face_state = ProgressState.COMPLETED
        sas.commit()
        history_ist.release_at = datetime.datetime.now()
        history_ist.result = 'success'
        sas.commit()


def delete_face(video_id: int):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        video_ist.face_state = ProgressState.IN_PROGRESS
        sas.commit()

        for face_ist in sas.exec(select(_Faces).filter(_Faces.video_id == video_id)).all():  # type: _Faces
            face_ist.preview_path.unlink(missing_ok=True)
            face_ist.sticker_path.unlink(missing_ok=True)
            sas.delete(face_ist)
            sas.commit()

        video_ist.face_state = ProgressState.NOT_STARTED
        sas.commit()


class VideoScene(t.NamedTuple):
    start_at: float
    frame: np.ndarray


def update_scene(video_id: int, scene_sequence: t.Iterable[VideoScene]):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        if video_ist.scene_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('scene block')
        else:
            video_ist.scene_state = ProgressState.IN_PROGRESS
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            history_ist = _SlaveHistory(video_id=video_id, processor='scene')
            sas.add(history_ist)
            sas.commit()
            for scene_record in scene_sequence:
                preview_path = _MINOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(preview_path.absolute().__str__(), scene_record.frame)
                scene_ist = _Scenes(
                    video_id=video_id,
                    history_id=history_ist.id,
                    start_at=scene_record.start_at,
                    preview_path=preview_path
                )
                sas.add(scene_ist)
                sas.commit()
        video_ist.scene_state = ProgressState.COMPLETED
        sas.commit()
        history_ist.release_at = datetime.datetime.now()
        history_ist.result = 'success'
        sas.commit()


def delete_scene(video_id: int):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        video_ist.scene_state = ProgressState.IN_PROGRESS
        sas.commit()
        for scene_ist in sas.exec(select(_Scenes).filter(_Scenes.video_id == video_id)).all():  # type: _Scenes
            scene_ist.preview_path.unlink(missing_ok=True)
            sas.delete(scene_ist)
            sas.commit()
        video_ist.scene_state = ProgressState.NOT_STARTED
        sas.commit()


class VideoBodyPart(t.NamedTuple):
    part: str
    ts: float
    frame: np.ndarray


def update_body_part(video_id: int, body_part_sequence: t.Iterable[VideoBodyPart]):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        if video_ist.body_part_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('body part block')
        else:
            video_ist.body_part_state = ProgressState.IN_PROGRESS
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            history_ist = _SlaveHistory(video_id=video_id, processor='body part')
            sas.add(history_ist)
            sas.commit()
            for body_part_record in body_part_sequence:
                frame_path = _MINOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(frame_path.absolute().__str__(), body_part_record.frame)
                body_part_ist = _BodyParts(
                    video_id=video_id,
                    history_id=history_ist.id,
                    part=body_part_record.part,
                    start_at=body_part_record.ts,
                    frame_path=frame_path
                )
                sas.add(body_part_ist)
                sas.commit()
        video_ist.body_part_state = ProgressState.COMPLETED
        sas.commit()
        history_ist.release_at = datetime.datetime.now()
        history_ist.result = 'success'
        sas.commit()


def delete_body_part(video_id: int):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        video_ist.body_part_state = ProgressState.IN_PROGRESS
        sas.commit()
        for body_part_ist in sas.exec(
                select(_BodyParts).filter(_BodyParts.video_id == video_id)).all():  # type: _BodyParts
            body_part_ist.frame_path.unlink(missing_ok=True)
            sas.delete(body_part_ist)
            sas.commit()
        video_ist.body_part_state = ProgressState.NOT_STARTED
        sas.commit()


def update_quick_look(video_id: int, quick_look_video: bytes):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        if video_ist.quick_look_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('quick look block')
        else:
            video_ist.quick_look_state = ProgressState.IN_PROGRESS
            sas.commit()
            history_ist = _SlaveHistory(video_id=video_id, processor='quick look')
            sas.add(history_ist)
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            quick_look_path = _MAJOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webm'
            with open(quick_look_path.absolute().__str__(), 'wb') as f:
                f.write(quick_look_video)
            quick_look_ist = _QuickLooks(
                video_id=video_id,
                history_id=history_ist.id,
                path=quick_look_path
            )
            sas.add(quick_look_ist)
            sas.commit()
        video_ist.quick_look_state = ProgressState.COMPLETED
        sas.commit()
        history_ist.release_at = datetime.datetime.now()
        history_ist.result = 'success'
        sas.commit()


def delete_quick_look(video_id: int):
    with _use_session() as sas:
        video_ist: Videos = sas.exec(select(Videos).filter(Videos.id == video_id)).one()
        video_ist.quick_look_state = ProgressState.IN_PROGRESS
        sas.commit()
        for quick_look_ist in sas.exec(
                select(_QuickLooks).filter(_QuickLooks.video_id == video_id)).all():  # type: _QuickLooks
            quick_look_ist.path.unlink(missing_ok=True)
            sas.delete(quick_look_ist)
            sas.commit()
        video_ist.quick_look_state = ProgressState.NOT_STARTED
        sas.commit()
