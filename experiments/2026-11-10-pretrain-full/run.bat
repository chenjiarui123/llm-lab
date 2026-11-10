@echo off
cd /d E:\3_llm\llm-lab
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
E:\ai\miniconda3\envs\torch\python.exe scripts\launch.py pretrain --config experiments\2026-11-10-pretrain-full\config.yaml > experiments\2026-11-10-pretrain-full\train.log 2>&1
echo === FINISHED rc=%errorlevel% at %date% %time% === >> experiments\2026-11-10-pretrain-full\train.log
