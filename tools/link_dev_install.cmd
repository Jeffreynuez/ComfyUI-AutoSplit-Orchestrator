@echo off
rem ---------------------------------------------------------------------------
rem  2D AutoSplit Studio - link this repo into ComfyUI and Krita
rem
rem  Replaces the copied node package in ComfyUI\custom_nodes and the copied
rem  plugin in Krita's pykrita folder with directory junctions that point at
rem  this repo, so ComfyUI and Krita always run the code in the repo and there
rem  is no second copy to forget. Existing copies are MOVED to a dated backup
rem  folder next to the ComfyUI install, never deleted. Junctions need no
rem  admin rights. Safe to run again.
rem
rem    tools\link_dev_install.cmd           link (close ComfyUI and Krita first)
rem    tools\link_dev_install.cmd unlink    remove the junctions only
rem
rem  The window waits for a key at the end so a double-click shows the result.
rem  Set AUTOSPLIT_NOPAUSE=1 to skip that when running it from a script.
rem
rem  ComfyUI root: AUTOSPLIT_COMFY_ROOT if set, else the Easy-Install layout
rem  beside the repo: <repo>\..\AI Work\ComfyUI-Easy-Install\ComfyUI
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
set "RC=0"

for %%I in ("%~dp0..") do set "REPO=%%~fI"
if defined AUTOSPLIT_COMFY_ROOT (
  set "COMFY=%AUTOSPLIT_COMFY_ROOT%"
) else (
  set "COMFY=%REPO%\..\AI Work\ComfyUI-Easy-Install\ComfyUI"
)
for %%I in ("%COMFY%") do set "COMFY=%%~fI"
set "PYKRITA=%APPDATA%\krita\pykrita"
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%T"
for %%I in ("%COMFY%\..") do set "BK=%%~fI\autosplit_link_backup\%STAMP%"

set "NODE_SRC=%REPO%\ComfyUI-AutoSplit-Orchestrator"
set "NODE_DST=%COMFY%\custom_nodes\ComfyUI-AutoSplit-Orchestrator"
set "KRITA_SRC=%REPO%\krita_plugin\autosplit_studio"
set "KRITA_DST=%PYKRITA%\autosplit_studio"

echo.
echo  repo     %REPO%
echo  ComfyUI  %COMFY%
echo  Krita    %PYKRITA%
echo.

if /I "%~1"=="unlink" goto :unlink

if not exist "%NODE_SRC%\nodes_sam3.py" (
  echo  ERROR: "%NODE_SRC%" does not look like the node package. Run this from the repo's tools folder.
  goto :fail
)
if not exist "%COMFY%\custom_nodes\" (
  echo  ERROR: no custom_nodes folder in "%COMFY%". Set AUTOSPLIT_COMFY_ROOT to your ComfyUI folder.
  goto :fail
)

call :link "%NODE_SRC%" "%NODE_DST%" "custom_nodes-ComfyUI-AutoSplit-Orchestrator" || goto :fail

if exist "%PYKRITA%\" (
  call :link "%KRITA_SRC%" "%KRITA_DST%" "pykrita-autosplit_studio" || goto :fail
  copy /Y "%REPO%\krita_plugin\autosplit_studio.desktop" "%PYKRITA%\" >nul
  echo   copied autosplit_studio.desktop into pykrita
) else (
  echo   skip Krita: "%PYKRITA%" not found ^(start Krita once, or ignore if you do not use the plugin^)
)

if not exist "%NODE_DST%\nodes_sam3.py" (
  echo   ERROR: the link was made but "%NODE_DST%\nodes_sam3.py" cannot be read through it.
  goto :fail
)

echo.
echo  Done. Restart ComfyUI and Krita so they load the linked code.
if exist "%BK%\" echo  Old copies are in "%BK%". Delete that folder once you are happy.
goto :done

:unlink
call :drop "%NODE_DST%"
call :drop "%KRITA_DST%"
echo.
echo  Junctions removed. Your earlier copies, if any, are under
for %%I in ("%COMFY%\..") do echo  "%%~fI\autosplit_link_backup"
goto :done

:fail
set "RC=1"
echo.
echo  Stopped. Nothing after the error above was changed.

:done
echo.
if not defined AUTOSPLIT_NOPAUSE pause
exit /b %RC%

rem ---- :link SRC DST LABEL --------------------------------------------------
:link
set "PARENT=%~dp2"
set "PARENT=%PARENT:~0,-1%"
set "NAME=%~nx2"
dir /AL /B "%PARENT%" 2>nul | findstr /X /I /C:"%NAME%" >nul
if not errorlevel 1 (
  rem already a junction: drop the link itself (rmdir without /s never touches the target)
  rmdir "%~2"
) else if exist "%~2\" (
  if not exist "%BK%\" mkdir "%BK%"
  move "%~2" "%BK%\%~3" >nul || (
    echo   ERROR: could not move "%~2". Close ComfyUI and Krita and run this again.
    exit /b 1
  )
  echo   moved the old copy to "%BK%\%~3"
)
mklink /J "%~2" "%~1" >nul || (
  echo   ERROR: mklink /J failed for "%~2"
  exit /b 1
)
echo   linked "%~2"
echo       -^> "%~1"
exit /b 0

rem ---- :drop DST: remove a junction, leave real folders alone ---------------
:drop
set "PARENT=%~dp1"
set "PARENT=%PARENT:~0,-1%"
set "NAME=%~nx1"
dir /AL /B "%PARENT%" 2>nul | findstr /X /I /C:"%NAME%" >nul
if not errorlevel 1 (
  rmdir "%~1" && echo   removed junction "%~1"
) else (
  echo   not a junction, left alone: "%~1"
)
exit /b 0
