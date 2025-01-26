import datetime
import mimetypes
import os
import typing as t
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, Request, HTTPException, APIRouter, Path
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

import src.dao as dao
from src.file_index import GUI_STORAGE
from src.utils import create_webp_b64

_API_ROUTER = APIRouter()


class PreviewRequest(BaseModel):
    offset: int
    limit: int = Field(..., le=24, gt=0, description='limit of video to show')
    filter: None | str = None


class PreviewItem(BaseModel):
    create_at: datetime.datetime
    video_pid: str
    face_pid: str
    quick_look_pid: str
    age: float
    duration: float
    name: str
    srt_ready: bool

    @field_validator('age')
    @classmethod
    def control_age(cls, v):
        if v < 0 or v > 100:
            return 0
        else:
            return v


@_API_ROUTER.post('/list-preview', response_model=list[PreviewItem])
def _list_preview(body: PreviewRequest) -> list[PreviewItem]:
    finish_videos = dao.page_videos_with_state(body.limit, body.offset, dao.ProgressState.COMPLETED, body.filter)

    if not finish_videos:
        return []

    else:
        video_id_list = [v.id for v in finish_videos]
        res_ls = []
        face_mapping: t.DefaultDict[int, list[dao.VideoFaces]] = defaultdict(list)
        for face_record in dao.query_face_by_video(video_id_list):
            face_mapping[face_record.video_id].append(face_record)

        quick_mapping: t.DefaultDict[int, list[dao.VideoQuickLooks]] = defaultdict(list)
        for quick_record in dao.query_quick_look_by_video(video_id_list):
            quick_mapping[quick_record.video_id].append(quick_record)

        for video_record in finish_videos:
            age_value = 0.0
            face_pid_value = ''
            quick_pid_value = ''

            if face_ls := face_mapping[video_record.id]:
                age_value = face_ls[0].age
                face_pid_value = face_ls[0].pid

            if quick_ls := quick_mapping[video_record.id]:
                quick_pid_value = quick_ls[0].pid

            item = PreviewItem(
                video_pid=video_record.pid,
                face_pid=face_pid_value,
                quick_look_pid=quick_pid_value,
                age=age_value,
                duration=video_record.file_duration_in_second,
                name=video_record.file_path.with_suffix('').name,
                srt_ready=dao.calculate_audio_progress_state(video_record) == dao.ProgressState.COMPLETED,
                create_at=video_record.file_create_at
            )

            res_ls.append(item)
        return res_ls


class VideoTsRequest(BaseModel):
    video_pid: str


class VideoTsItem(BaseModel):
    ts: float
    preview: str
    reason: str


class SrtText(BaseModel):
    lang: str
    text: str


class SrtItem(BaseModel):
    start_at: float
    end_at: float
    texts: list[SrtText]


class VideoTsResponse(BaseModel):
    ts_list: list[VideoTsItem]
    srt_list: list[SrtItem]
    video_name: str
    video_create_at: datetime.datetime


@_API_ROUTER.post('/video-ts', response_model=VideoTsResponse)
def _video_ts(body: VideoTsRequest) -> VideoTsResponse:
    video_ist = dao.query_video_by_pid(body.video_pid)
    if not video_ist:
        raise HTTPException(status_code=404)

    res_maybe_repeat = []
    for scene_ist in dao.query_scene_by_video([video_ist.id]):
        res_maybe_repeat.append(
            VideoTsItem(ts=scene_ist.start_at, preview=create_webp_b64(scene_ist.preview_path), reason='scene')
        )

    for body_ist in dao.query_body_part_by_video([video_ist.id]):
        res_maybe_repeat.append(
            VideoTsItem(ts=body_ist.start_at, preview=create_webp_b64(body_ist.frame_path), reason=body_ist.part)
        )

    face_list = dao.query_face_by_video([video_ist.id])
    for face_ist in face_list:
        res_maybe_repeat.append(
            VideoTsItem(ts=0, preview=create_webp_b64(face_ist.preview_path), reason='face')
        )

    # 每10秒最多一帧
    mapping = defaultdict(list)
    for r in res_maybe_repeat:
        mapping[int(r.ts) // 10].append(r)

    seek_ls = [v_ls[0] for k, v_ls in mapping.items()]
    seek_ls.sort(key=lambda x: x.ts)

    srt_ls = []
    srt_dao_ls = dao.query_srt_by_video([video_ist.id])
    for srt_ist in srt_dao_ls:
        text_list = []
        if srt_ist.asr_text:
            text_list.append(SrtText(lang='ja-JP', text=srt_ist.asr_text))
        if srt_ist.translate_text:
            text_list.append(SrtText(lang='zh-CN', text=srt_ist.translate_text))
        srt_ls.append(
            SrtItem(
                start_at=srt_ist.start_at,
                end_at=srt_ist.end_at,
                texts=text_list
            )
        )
    srt_ls.sort(key=lambda x: x.start_at)

    return VideoTsResponse(ts_list=seek_ls, srt_list=srt_ls, video_name=video_ist.file_path.with_suffix('').name,
                           video_create_at=video_ist.file_create_at)


@_API_ROUTER.get('/face-image/{path}', response_class=FileResponse)
def _face_image(path: str = Path(...)) -> FileResponse:
    record = dao.query_face_by_pid(path)
    if record:
        return FileResponse(record.sticker_path)
    else:
        raise HTTPException(status_code=404)


@_API_ROUTER.get('/quick-look/{path}', response_class=FileResponse)
def _quick_look(path: str = Path(...)) -> FileResponse:
    record = dao.query_quick_look_by_pid(path)
    if record:
        return FileResponse(record.path)
    else:
        raise HTTPException(status_code=404)


@_API_ROUTER.get('/video-full/{path}', response_class=FileResponse)
def _video_full(path: str = Path(...)) -> FileResponse:
    record = dao.query_video_by_pid(path)
    if record.file_path.exists():
        return FileResponse(record.file_path)
    else:
        raise HTTPException(status_code=404)


APP = FastAPI()

mimetypes.add_type('application/javascript', '.js')
APP.include_router(_API_ROUTER, prefix='/api')


@APP.get('/', response_class=FileResponse)
def _index() -> FileResponse:
    return FileResponse(GUI_STORAGE / 'index.html', headers={'Content-Type': 'text/html'})


@APP.get("/{full_path:path}")
async def catch_all(full_path: str):
    static_file_path = GUI_STORAGE / full_path
    if os.path.isfile(static_file_path):
        return FileResponse(static_file_path)
    else:
        raise HTTPException(status_code=404)


@APP.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 403:
        return JSONResponse(
            status_code=exc.status_code,
            content={'message': exc.detail},
        )
    elif exc.status_code == 404:
        return FileResponse(GUI_STORAGE / 'index.html')


if __name__ == '__main__':
    uvicorn.run(APP, host='0.0.0.0', port=8000)
