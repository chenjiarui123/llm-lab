# Windows 环境初始化（PowerShell）
# 用法： powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
#       加 -Reinstall 强制重建 venv
#       加 -NoTorch  跳过 torch 检查（默认只检查不安装）

param(
    [switch]$Reinstall,
    [switch]$NoTorch
)

$ErrorActionPreference = "Stop"

# 切到 llm-lab 根目录（脚本的上一级）
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "==> 工作目录: $Root" -ForegroundColor Cyan

# ---------- 1. 检查 Python ----------
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "找不到 python，请先安装 Python 3.10+ 并加入 PATH" }
python --version

# ---------- 2. 创建 venv（复用系统 torch）----------
$venv = Join-Path $Root ".venv"
if ($Reinstall -and (Test-Path $venv)) {
    Write-Host "==> 删除旧 venv" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $venv)) {
    Write-Host "==> 创建 venv (--system-site-packages，复用已装好的 torch)" -ForegroundColor Cyan
    python -m venv .venv --system-site-packages
}

$pyExe = Join-Path $venv "Scripts\python.exe"
& $pyExe -m pip install --upgrade pip -q

# ---------- 3. 安装依赖 ----------
Write-Host "==> 安装 requirements/base.txt" -ForegroundColor Cyan
& $pyExe -m pip install -r requirements/base.txt

Write-Host "==> 安装 requirements/train.txt" -ForegroundColor Cyan
& $pyExe -m pip install -r requirements/train.txt

# ---------- 4. 检查 torch / CUDA ----------
if (-not $NoTorch) {
    Write-Host "==> 检查 torch" -ForegroundColor Cyan
    & $pyExe -c @"
import torch
print('torch      :', torch.__version__)
print('cuda avail :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu        :', torch.cuda.get_device_name(0))
    print('vram       :', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')
    print('bf16       :', torch.cuda.is_bf16_supported())
"@
}

# ---------- 5. clone MiniMind ----------
$mm = Join-Path $Root "third_party/minimind"
if (-not (Test-Path $mm)) {
    Write-Host "==> clone MiniMind -> third_party/minimind" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "third_party") | Out-Null
    git clone --depth 1 https://github.com/jingyaogong/minimind $mm
} else {
    Write-Host "==> third_party/minimind 已存在，跳过 clone" -ForegroundColor Yellow
}

Push-Location $mm
$commit = (git rev-parse --short HEAD)
Pop-Location
Write-Host "==> MiniMind commit: $commit" -ForegroundColor Green

# ---------- 6. 目录 ----------
foreach ($d in @("dataset", "out")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
}

Write-Host ""
Write-Host "完成。下一步：" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\download_data.py --mini"
Write-Host "  python scripts\launch.py pretrain --config configs\local_8gb.yaml --dry-run"
