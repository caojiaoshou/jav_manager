cd %~dp0
call .venv\Scripts\activate.bat
set PYTHONPATH=%CD%
.venv\Scripts\python.exe src\master.py
pause
