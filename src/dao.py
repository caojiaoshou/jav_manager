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
from sqlmodel import SQLModel, Field, create_engine, Session, select, all_, any_

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


class VideoInfo(SQLModel, table=True):
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
    pid: str = Field(index=True, default_factory=lambda: uuid.uuid4().hex)
    file_duration_in_second: float = Field(default=0)
    record_add_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    file_create_at: datetime.datetime
    hash: str
    size: int


class VideoFaces(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    embedding: np.ndarray = Field(sa_column=Column(Float32ArrayType))
    preview_path: pathlib.Path = Field(sa_column=Column(PathType))
    sticker_path: pathlib.Path = Field(sa_column=Column(PathType))
    age: float
    pid: str = Field(index=True, default_factory=lambda: uuid.uuid4().hex)

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )


class VideoQuickLooks(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    path: pathlib.Path = Field(sa_column=Column(PathType))
    pid: str = Field(index=True, default_factory=lambda: uuid.uuid4().hex)


class VideoSpeeches(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    start_at: float
    end_at: float
    asr_text: str
    translate_text: str
    pid: str = Field(index=True, default_factory=lambda: uuid.uuid4().hex)


class VideoScenes(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    start_at: float
    preview_path: pathlib.Path = Field(sa_column=Column(PathType))
    pid: str = Field(index=True, default_factory=lambda: uuid.uuid4().hex)


class VideoBodyParts(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    history_id: int = Field(index=True)
    part: str
    start_at: float
    frame_path: pathlib.Path = Field(sa_column=Column(PathType))
    pid: str = Field(index=True, default_factory=lambda: uuid.uuid4().hex)


class SlaveHistory(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    video_id: int = Field(index=True)
    acquire_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    release_at: datetime.datetime = Field(default=datetime.datetime.min)
    processor: str = Field(default='')
    result: str = Field(default='')
    pid: str = Field(index=True, default_factory=lambda: uuid.uuid4().hex)


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


def add_video(group: str, file_path: pathlib.Path, file_create_at: datetime.datetime, file_duration_in_second: float,
              hash: str, size: int):
    with _use_session() as session:
        session.add(
            VideoInfo(group=group, file_path=file_path, file_create_at=file_create_at,
                      file_duration_in_second=file_duration_in_second, hash=hash, size=size)
        )
        session.commit()


def delete_video(video_id: int):
    with _use_session() as session:
        ist = session.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        ist.delete = True
        session.commit()


def page_videos_with_state(
        limit: int | None,
        offset: int | None,
        video_state: ProgressState | None
) -> list[VideoInfo]:
    filters = [VideoInfo.delete == False]
    match video_state:
        case ProgressState.IN_PROGRESS:
            filters.append(
                any_(
                    VideoInfo.face_state == ProgressState.IN_PROGRESS,
                    VideoInfo.scene_state == ProgressState.IN_PROGRESS,
                    VideoInfo.body_part_state == ProgressState.IN_PROGRESS,
                    VideoInfo.quick_look_state == ProgressState.IN_PROGRESS,
                )
            )
        case ProgressState.NOT_STARTED:
            filters.append(
                all_(
                    VideoInfo.face_state == ProgressState.NOT_STARTED,
                    VideoInfo.scene_state == ProgressState.NOT_STARTED,
                    VideoInfo.body_part_state == ProgressState.NOT_STARTED,
                    VideoInfo.quick_look_state == ProgressState.NOT_STARTED,
                )
            )
        case ProgressState.COMPLETED:
            filters.append(
                all_(
                    VideoInfo.face_state == ProgressState.COMPLETED,
                    VideoInfo.scene_state == ProgressState.COMPLETED,
                    VideoInfo.body_part_state == ProgressState.COMPLETED,
                    VideoInfo.quick_look_state == ProgressState.COMPLETED,
                )
            )
        case None:
            ...
        case _:
            raise ValueError(f'{video_state=}')

    query = select(VideoInfo).filter(*filters)

    if limit is not None:
        query = query.limit(limit)
    if offset is not None:
        query = query.offset(offset)

    with _use_session() as session:
        rt = session.exec(query).all()
    return rt


def query_video(video_id_seq: t.Sequence[int]) -> list[VideoInfo]:
    with _use_session() as session:
        rt = session.exec(select(VideoInfo).filter(VideoInfo.id.in_(video_id_seq))).all()
    return rt


def list_videos() -> list[VideoInfo]:
    with _use_session() as session:
        rt = session.exec(select(VideoInfo).filter(VideoInfo.delete == False)).all()
    return rt


class VideoFaceParams(t.NamedTuple):
    embedding: np.ndarray
    age: float
    frame: np.ndarray
    crop_image: np.ndarray


def update_face(video_id: int, face_sequence: t.Iterable[VideoFaceParams]):
    with _use_session() as sas:
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        if video_ist.face_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('face block')
        else:
            video_ist.face_state = ProgressState.IN_PROGRESS
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            history_ist = SlaveHistory(video_id=video_id, processor='face')
            sas.add(history_ist)
            sas.commit()
            for face_record in face_sequence:
                sticker_path = _MAJOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(sticker_path.absolute().__str__(), face_record.crop_image)
                raw_path = _MINOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(raw_path.absolute().__str__(), face_record.frame)
                face_ist = VideoFaces(
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
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        video_ist.face_state = ProgressState.IN_PROGRESS
        sas.commit()

        for face_ist in sas.exec(select(VideoFaces).filter(VideoFaces.video_id == video_id)).all():  # type: VideoFaces
            face_ist.preview_path.unlink(missing_ok=True)
            face_ist.sticker_path.unlink(missing_ok=True)
            sas.delete(face_ist)
            sas.commit()

        video_ist.face_state = ProgressState.NOT_STARTED
        sas.commit()


def query_face_by_video(video_id_seq: t.Sequence[int]) -> list[VideoFaces]:
    with _use_session() as sas:
        rt = sas.exec(select(VideoFaces).filter(VideoFaces.video_id.in_(video_id_seq))).all()
        return rt


def query_face_by_db_id(face_id: int) -> list[VideoFaces]:
    with _use_session() as sas:
        rt = sas.exec(select(VideoFaces).filter(VideoFaces.id == face_id)).one()
        return rt


def query_face_by_pid(face_pid: str) -> VideoFaces:
    with _use_session() as sas:
        rt = sas.exec(select(VideoFaces).filter(VideoFaces.pid == face_pid)).one()
        return rt


class VideoSceneParams(t.NamedTuple):
    start_at: float
    frame: np.ndarray


def update_scene(video_id: int, scene_sequence: t.Iterable[VideoSceneParams]):
    with _use_session() as sas:
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        if video_ist.scene_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('scene block')
        else:
            video_ist.scene_state = ProgressState.IN_PROGRESS
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            history_ist = SlaveHistory(video_id=video_id, processor='scene')
            sas.add(history_ist)
            sas.commit()
            for scene_record in scene_sequence:
                preview_path = _MINOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(preview_path.absolute().__str__(), scene_record.frame)
                scene_ist = VideoScenes(
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
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        video_ist.scene_state = ProgressState.IN_PROGRESS
        sas.commit()
        for scene_ist in sas.exec(
                select(VideoScenes).filter(VideoScenes.video_id == video_id)).all():  # type: VideoScenes
            scene_ist.preview_path.unlink(missing_ok=True)
            sas.delete(scene_ist)
            sas.commit()
        video_ist.scene_state = ProgressState.NOT_STARTED
        sas.commit()


def query_scene_by_video(video_id_seq: t.Sequence[int]) -> list[VideoScenes]:
    with _use_session() as sas:
        rt = sas.exec(select(VideoScenes).filter(VideoScenes.video_id.in_(video_id_seq))).all()
        return rt


def query_scene_by_db_id(scene_id: int) -> list[VideoScenes]:
    with _use_session() as sas:
        rt = sas.exec(select(VideoScenes).filter(VideoScenes.id == scene_id)).one()
        return rt


def query_scene_by_pid(scene_pid: str) -> VideoScenes:
    with _use_session() as sas:
        rt = sas.exec(select(VideoScenes).filter(VideoScenes.pid == scene_pid)).one()
        return rt


class VideoBodyPartParams(t.NamedTuple):
    part: str
    ts: float
    frame: np.ndarray


def update_body_part(video_id: int, body_part_sequence: t.Iterable[VideoBodyPartParams]):
    with _use_session() as sas:
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        if video_ist.body_part_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('body part block')
        else:
            video_ist.body_part_state = ProgressState.IN_PROGRESS
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            history_ist = SlaveHistory(video_id=video_id, processor='body part')
            sas.add(history_ist)
            sas.commit()
            for body_part_record in body_part_sequence:
                frame_path = _MINOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webp'
                cv2.imwrite(frame_path.absolute().__str__(), body_part_record.frame)
                body_part_ist = VideoBodyParts(
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
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        video_ist.body_part_state = ProgressState.IN_PROGRESS
        sas.commit()
        for body_part_ist in sas.exec(
                select(VideoBodyParts).filter(VideoBodyParts.video_id == video_id)).all():  # type: VideoBodyParts
            body_part_ist.frame_path.unlink(missing_ok=True)
            sas.delete(body_part_ist)
            sas.commit()
        video_ist.body_part_state = ProgressState.NOT_STARTED
        sas.commit()


def query_body_part_by_video(video_id_seq: t.Sequence[int]) -> list[VideoBodyParts]:
    with _use_session() as sas:
        rt = sas.exec(select(VideoBodyParts).filter(VideoBodyParts.video_id.in_(video_id_seq))).all()
        return rt


def query_body_part_by_db_id(body_part_id: int) -> VideoBodyParts:
    with _use_session() as sas:
        rt = sas.exec(select(VideoBodyParts).filter(VideoBodyParts.id == body_part_id)).one()
        return rt


def query_body_part_by_pid(body_part_pid: str) -> VideoBodyParts:
    with _use_session() as sas:
        rt = sas.exec(select(VideoBodyParts).filter(VideoBodyParts.pid == body_part_pid)).one()
        return rt


def update_quick_look(video_id: int, quick_look_video: bytes):
    with _use_session() as sas:
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        if video_ist.quick_look_state == ProgressState.IN_PROGRESS:
            raise RacingProgressError('quick look block')
        else:
            video_ist.quick_look_state = ProgressState.IN_PROGRESS
            sas.commit()
            history_ist = SlaveHistory(video_id=video_id, processor='quick look')
            sas.add(history_ist)
            sas.commit()
            video_name = video_ist.file_path.with_suffix('').name
            quick_look_path = _MAJOR_STORAGE / f'{video_name}_{uuid.uuid4().hex}.webm'
            with open(quick_look_path.absolute().__str__(), 'wb') as f:
                f.write(quick_look_video)
            quick_look_ist = VideoQuickLooks(
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
        video_ist: VideoInfo = sas.exec(select(VideoInfo).filter(VideoInfo.id == video_id)).one()
        video_ist.quick_look_state = ProgressState.IN_PROGRESS
        sas.commit()
        for quick_look_ist in sas.exec(
                select(VideoQuickLooks).filter(VideoQuickLooks.video_id == video_id)).all():  # type: VideoQuickLooks
            quick_look_ist.path.unlink(missing_ok=True)
            sas.delete(quick_look_ist)
            sas.commit()
        video_ist.quick_look_state = ProgressState.NOT_STARTED
        sas.commit()


def query_quick_look_by_video(video_id_seq: t.Sequence[int]) -> list[VideoQuickLooks]:
    with _use_session() as sas:
        rt = sas.exec(select(VideoQuickLooks).filter(VideoQuickLooks.video_id.in_(video_id_seq))).all()
        return rt


def query_quick_look_by_db_id(quick_look_id: int) -> VideoQuickLooks:
    with _use_session() as sas:
        rt = sas.exec(select(VideoQuickLooks).filter(VideoQuickLooks.id == quick_look_id)).one()
        return rt


def query_quick_look_by_pid(quick_look_pid: str) -> VideoQuickLooks:
    with _use_session() as sas:
        rt = sas.exec(select(VideoQuickLooks).filter(VideoQuickLooks.pid == quick_look_pid)).one()
        return rt


def calculate_video_progress_state(video: VideoInfo) -> ProgressState:
    states = video.face_state, video.body_part_state, video.scene_state, video.quick_look_state
    if all(map(lambda x: x == ProgressState.NOT_STARTED, states)):
        return ProgressState.NOT_STARTED
    elif all(map(lambda x: x == ProgressState.COMPLETED, states)):
        return ProgressState.COMPLETED
    else:
        return ProgressState.IN_PROGRESS


if __name__ == '__main__':
    print(list_videos())
