@echo off
REM Batch-convert a folder of voice lines with one RVC model.
REM Edit the settings below, then double-click this file
REM (or drag a folder of voice lines onto it).
cd /d "%~dp0"

REM ================= Settings =================
REM Model folder name inside models\ (run once with MODEL empty to see the list)
set "MODEL="
REM Folder with the source voice lines (a folder dropped on this .bat overrides it)
set "INPUT=input"
REM Output folder (empty = output\<model name>)
set "OUTPUT="
REM Transpose in semitones, e.g. 12 = one octave up, -12 = one octave down
set "PITCH=0"
REM Pitch method: rmvpe (recommended), hybrid, harvest, crepe, crepe-tiny, dio, pm
set "METHOD=rmvpe"
REM Quality: max (fp32, RMVPE+crepe pitch, wider index search, bigger chunks) or normal (faster)
set "QUALITY=max"
REM Feature retrieval rate 0.0 - 1.0
set "INDEX_RATE=0.4"
REM Protect consonants/breaths 0.0 - 0.5 (0.5 = off)
set "PROTECT=0.33"
REM Volume envelope mix 0.0 - 1.0 (1 = off, lower = follow the input's loudness)
set "RMS_MIX=1"
REM Octave guard: down (fix octave drops), both, off
set "OCTAVE_FIX=down"
REM Output sample rate: model, source (match each input file), or a number like 44100
set "SR=model"
REM =============================================

if not "%~1"=="" set "INPUT=%~1"

set "PY=runtime\python.exe"
if not exist "%PY%" set "PY=python"

if "%MODEL%"=="" (
    "%PY%" rvc_batch.py --list-models
    echo.
    echo Set MODEL at the top of RVC-Batch.bat to one of the names above.
    pause
    exit /b
)

if not exist "%INPUT%" (
    mkdir "%INPUT%"
    echo Created the folder "%INPUT%" - put your voice lines in it and run this again.
    pause
    exit /b
)

set ARGS=--model "%MODEL%" --input "%INPUT%" --pitch %PITCH% --method %METHOD% --quality %QUALITY% --index-rate %INDEX_RATE% --protect %PROTECT% --rms-mix %RMS_MIX% --octave-fix %OCTAVE_FIX% --sr %SR%
if not "%OUTPUT%"=="" set ARGS=%ARGS% --output "%OUTPUT%"

"%PY%" rvc_batch.py %ARGS%
pause
