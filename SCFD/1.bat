@echo off
REM Activate the virtual environment
call activate myclone
REM Define an array of folder names
set FOLDERS="brigh_cont" "brigh_cont - rotate" "brigh_gray_rotate" "gray_scale" "gray_scale - rotation" "main_rotation"

REM Iterate over each folder and run the command
for %%f in (%FOLDERS%) do (
    echo Processing folder: %%f
    python demo\image_demo.py %%f configs\scrfd\scrfd_34g.py scrfd_34g.pth res_%%f
    echo Finished processing folder: %%f
    echo.
)
REM Deactivate the virtual environment when done
CALL deactivate
echo All folders processed.
