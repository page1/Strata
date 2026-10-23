@echo off
rem tools/vision/strata-vision-npu.cmd - Windows launcher for the Intel NPU picture reader.
rem
rem A run config's vision section names one exe and passes only the flags of the mtmd readers, so on Windows the
rem section should point at this file (a .py is not a program CreateProcess can start).  Everything is forwarded
rem to the Python reader; its own settings come from the environment: STRATA_VISION_NPU_MODEL_IR,
rem STRATA_VISION_NPU_DEVICE, STRATA_VISION_NPU_MOCK (see the docstring of strata_vision_npu.py, or docs/INTEL_NPU.md).
setlocal
set "PY=py -3"
where py >nul 2>nul || set "PY=python"
%PY% "%~dp0strata_vision_npu.py" %*
exit /b %ERRORLEVEL%
