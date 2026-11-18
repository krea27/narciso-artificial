@echo off
title NARCISO ARTIFICIAL - Prototipo Generativo
echo Iniciando Narciso Artificial...

if exist "C:\Users\dedic\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    "C:\Users\dedic\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run "%~dp0app.py" --server.headless=false
) else (
    python -m pip install -r "%~dp0requirements.txt"
    python -m streamlit run "%~dp0app.py" --server.headless=false
)
pause
