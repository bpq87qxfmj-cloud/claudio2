@echo off
rem Lo ejecuta el Programador de tareas de Windows. Para crearlo: python agente.py --instalar-windows
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist registro_agente mkdir registro_agente
python agente.py --una-vez %* >> registro_agente\consola.txt 2>&1
