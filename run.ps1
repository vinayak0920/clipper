# One command setup + start on Windows. Opens the dashboard at http://localhost:8000
# Run it with run.bat (double-click, or type run.bat in a terminal in this folder).
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot

function Refresh-Path {
  # winget installs update PATH in the registry, not in this window, so read it back.
  $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
  $user = [Environment]::GetEnvironmentVariable("Path", "User")
  $env:Path = "$machine;$user"
}

function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Stop-With($msg) {
  Write-Host ""
  Write-Host $msg -ForegroundColor Red
  Write-Host ""
  Read-Host "Press Enter to close"
  exit 1
}

function Has-Subtitles {
  if (-not (Has "ffmpeg")) { return $false }
  $filters = & ffmpeg -hide_banner -filters 2>$null | Out-String
  return $filters -match "\ssubtitles\s"
}

# ---------------------------------------------------------------- FFmpeg (with subtitles / libass)
if (-not (Has-Subtitles)) {
  if (Has "winget") {
    Write-Host "Installing FFmpeg with subtitle support (one time)..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Refresh-Path
  }
  if (-not (Has-Subtitles)) {
    Stop-With "FFmpeg with the subtitles filter is needed. Install it with: winget install Gyan.FFmpeg`nThen close this window and run run.bat again."
  }
}

# ---------------------------------------------------------------- Python 3.10 - 3.13
function Find-Python {
  if (Has "py") {
    foreach ($v in "3.12", "3.11", "3.13", "3.10") {
      $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
      if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
  }
  foreach ($c in Get-Command python, python3 -All -ErrorAction SilentlyContinue) {
    if ($c.Source -like "*WindowsApps*") { continue }   # the Microsoft Store stub, not a real Python
    $ver = & $c.Source -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
    if ($ver -in "3.10", "3.11", "3.12", "3.13") { return $c.Source }
  }
  return $null
}

$python = Find-Python
if (-not $python) {
  if (Has "winget") {
    Write-Host "Installing Python 3.12 (one time)..."
    winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
    Refresh-Path
    $python = Find-Python
  }
  if (-not $python) {
    Stop-With "Python 3.10 - 3.13 is needed. Install Python 3.12 from python.org (tick 'Add python.exe to PATH'),`nthen close this window and run run.bat again."
  }
}

# ---------------------------------------------------------------- virtual environment
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path ".venv") {
  $ver = if (Test-Path $venvPy) { & $venvPy -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null } else { "" }
  if ($ver -notin "3.10", "3.11", "3.12", "3.13") {
    $backup = ".venv-backup-" + (Get-Date -Format "yyyyMMddHHmmss")
    Write-Host "Moving unsupported or broken .venv (Python $ver) to $backup"
    Rename-Item ".venv" $backup
  }
}
if (-not (Test-Path ".venv")) {
  Write-Host "Creating Python environment with $python..."
  & $python -m venv .venv
  if ($LASTEXITCODE -ne 0) { Stop-With "Could not create the Python environment." }
}

$marker = ".venv\.installed"
if (-not (Test-Path $marker) -or (Get-Item "requirements.txt").LastWriteTime -gt (Get-Item $marker).LastWriteTime) {
  Write-Host "Installing Python packages (first run takes a few minutes)..."
  & $venvPy -m pip install -q --upgrade pip
  & $venvPy -m pip install -q -r requirements.txt
  if ($LASTEXITCODE -ne 0) { Stop-With "Installing the Python packages failed. Scroll up for the error." }
  New-Item -ItemType File -Force $marker | Out-Null
}
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

# streamlink lives in the environment's Scripts folder; the recorder calls it by name
$env:Path = (Join-Path $PSScriptRoot ".venv\Scripts") + ";" + $env:Path

# ---------------------------------------------------------------- Ollama (optional, free local AI)
if (Has "ollama") {
  try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://localhost:11434/api/tags" | Out-Null }
  catch { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden }
} else {
  Write-Host "Tip: for free local AI hooks, run: winget install Ollama.Ollama   then: ollama pull qwen2.5:7b"
}

# ---------------------------------------------------------------- start
$port = if ($env:PORT) { $env:PORT } else { "8000" }
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-Command", "Start-Sleep 3; Start-Process 'http://localhost:$port'"
Write-Host "Clipper is running at http://localhost:$port  (close this window to stop it)"
& $venvPy -m uvicorn clipper.server:app --host 127.0.0.1 --port $port
