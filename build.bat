@echo off
setlocal

set "VCVARS="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
)

if "%VCVARS%"=="" (
    echo Error: No se encontro vcvars64.bat de Visual Studio.
    exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b 1

if not exist build mkdir build

echo Compilando src\seird_basic.cu (sm_120)...
nvcc -std=c++17 -arch=sm_120 -O3 -Xcompiler "/EHsc /O2" src\seird_basic.cu -o build\seird_basic.exe -luser32 -lgdi32
if errorlevel 1 exit /b 1

echo Compilacion exitosa: build\seird_basic.exe
