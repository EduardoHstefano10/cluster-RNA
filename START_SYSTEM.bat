@echo off
REM Activar venv si existe
if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else if exist .\venv\Scripts\activate (
    call .\venv\Scripts\activate
)
REM Arrancar servidor (recomendado usar main.py)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
