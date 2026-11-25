@echo off
REM Compila RN1.tex con Tectonic (https://tectonic-typesetting.github.io).
REM Tambien se puede subir esta carpeta tal cual a Overleaf.
cd /d "%~dp0"
set TEC=tectonic
if exist tectonic.exe set TEC=tectonic.exe
where %TEC% >nul 2>nul
if errorlevel 1 (
  echo No se encontro Tectonic. Copia tectonic.exe en esta carpeta o instalalo desde
  echo https://tectonic-typesetting.github.io
  pause
  exit /b 1
)
%TEC% RN1.tex
pause
