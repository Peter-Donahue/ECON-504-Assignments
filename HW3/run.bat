@echo off

REM Ensure the Julia executable is in your PATH, otherwise
REM must provide full path.
set JULIA_EXEC=julia

REM ====================================
REM Task 1
REM ===================================
set SCRIPT_PATH=.\task1.jl
echo Running Task 1
%JULIA_EXEC% %SCRIPT_PATH%
echo Task 1 Complete

REM ====================================
REM Task 2
REM ===================================
set SCRIPT_PATH=.\task2.jl
set PARMS_PATH=.\parms.json
echo Running Task 2
%JULIA_EXEC% %SCRIPT_PATH% %PARMS_PATH%
echo Task 2 Complete

REM ====================================
REM Task 3 - 5
REM ===================================
set SCRIPT_PATH=.\task3.jl
echo Running Task 3-5
REM %JULIA_EXEC% %SCRIPT_PATH% %PARMS_PATH%
echo Task 3-5 Complete

REM Exit with the same status code as the Julia script
exit /b %errorlevel%
