@echo off
echo ==================================================
echo       Plex AI Dubbing Pipeline (E: Drive Setup)     
echo ==================================================
echo.

REM Activate the Miniconda environment safely
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" local-dubber

REM Prompt for user manual variables
set /p movie_file="1. Drag and drop or type path to input movie file: "
set /p voice_ref="2. Drag and drop or type path to 10s voice sample (.wav): "
set /p lang_choice="3. Enter target language (Telugu or Hindi): "
set /p output_file="4. Enter the full output path (e.g., E:\Movies\Output.mkv): "

echo.
echo [!] Starting processing pipeline... Please monitor the phases below.
echo.

python E:\local-dubber\dubber.py --input %movie_file% --voice %voice_ref% --lang %lang_choice% --output %output_file%

echo.
echo ==================================================
echo Task Processing Window Complete.
pause
