@echo off
REM Anaconda 환경 활성화
call C:\Users\dohyeon\anaconda3\Scripts\activate.bat maup

REM 첫 번째 Python 스크립트 실행
python D:\project_repository\flow_prediction\dockless\experiment\modules\revision\kansas\kansas_new_main_indi_grids_fixed_250m_test_0812.py

REM 두 번째 Python 스크립트 실행
python D:\project_repository\flow_prediction\dockless\experiment\modules\revision\kansas\kansas_new_main_indi_grids_fixed_750m_test_0812.py

REM Anaconda 환경 비활성화
conda deactivate

@echo on