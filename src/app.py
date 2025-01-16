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

_API_ROUTER = APIRouter()


class PreviewRequest(BaseModel):
    offset: int
    limit: int = Field(..., le=24, gt=0, description='limit of video to show')


class PreviewItem(BaseModel):
    video_pid: str
    face_pid: str
    quick_look_pid: str
    age: float
    duration: float
    name: str

    @field_validator('age')
    @classmethod
    def control_age(cls, v):
        if v < 0 or v > 100:
            return 0
        else:
            return v


@_API_ROUTER.post('/list-preview', response_model=list[PreviewItem])
def _list_preview(body: PreviewRequest) -> list[PreviewItem]:
    finish_videos = dao.page_videos_with_state(body.limit, body.offset, dao.ProgressState.COMPLETED)

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
                name=video_record.file_path.with_suffix('').name
            )

            res_ls.append(item)
        return res_ls


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
        return FileResponse(GUI_STORAGE / '404.html')


if __name__ == '__main__':
    uvicorn.run(APP, host='0.0.0.0', port=8000)
