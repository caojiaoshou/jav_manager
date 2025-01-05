import mimetypes
import os

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from src.file_index import GUI_STORAGE

APP = FastAPI()

mimetypes.add_type('application/javascript', '.js')


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
