<#
.SYNOPSIS
    Automates setup, dataset conversion, training, and the Streamlit demo
    for Human-Posture-CV.

.DESCRIPTION
    Flags:
      -All      Run install, convert, train, and launch the app (does NOT push)
      -Install  Create venv and install requirements.txt
      -Convert  Convert COCO annotations to YOLO labels for train/valid/test
      -Train    Run train.py
      -App      Launch the Streamlit demo
      -Push     Commit and push changes to git (skipped unless explicitly requested)
      -Epochs   Number of training epochs (default 50)
      -Batch    Training batch size (default 16)
      -Model    Pretrained backbone (default yolov8n.pt)
      -Repo     Git remote URL to push to, if -Push is used

.EXAMPLE
    .\setup_and_run.ps1 -All -Epochs 50 -Batch 16

.NOTES
    Security: this script never stores or hardcodes a token. If you use -Push,
    set $env:GITHUB_TOKEN yourself beforehand for non-interactive auth, or let
    git prompt you interactively. Never commit a token to the repository.
#>

param(
    [switch]$All,
    [switch]$Install,
    [switch]$Convert,
    [switch]$Train,
    [switch]$App,
    [switch]$Push,
    [int]$Epochs = 50,
    [int]$Batch = 16,
    [string]$Model = "yolov8n.pt",
    [string]$Repo = ""
)

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

if ($All) {
    $Install = $true
    $Convert = $true
    $Train = $true
    $App = $true
    # Note: -All intentionally does NOT push. Pass -Push explicitly if you want that.
}

if ($Install) {
    Step "Creating virtual environment (venv)"
    if (-not (Test-Path ".\venv")) {
        python -m venv venv
    }
    Step "Activating venv and installing requirements"
    & .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
}

if ($Convert) {
    Step "Converting COCO annotations to YOLO labels"
    python utils/convert_coco_to_yolo.py --coco images/train/_annotations.coco.json --images images/train --out labels/train
    python utils/convert_coco_to_yolo.py --coco images/valid/_annotations.coco.json --images images/valid --out labels/valid
    python utils/convert_coco_to_yolo.py --coco images/test/_annotations.coco.json  --images images/test  --out labels/test
}

if ($Train) {
    Step "Training model (epochs=$Epochs, batch=$Batch, backbone=$Model)"
    python train.py --data dataset.yaml --epochs $Epochs --batch $Batch --model $Model
}

if ($App) {
    Step "Launching Streamlit demo (Ctrl+C to stop)"
    streamlit run app.py
}

if ($Push) {
    Step "Committing and pushing changes"

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Error "git is not installed or not on PATH. Install Git and re-run with -Push."
        exit 1
    }

    if (-not (git config user.name)) {
        git config user.name "Human-Posture-CV Bot"
    }
    if (-not (git config user.email)) {
        git config user.email "bot@example.com"
    }

    git add -A
    git commit -m "Automated update via setup_and_run.ps1" --allow-empty-message 2>$null

    if ($Repo -ne "") {
        if (-not (git remote | Select-String "origin")) {
            git remote add origin $Repo
        } else {
            git remote set-url origin $Repo
        }
    }

    if ($env:GITHUB_TOKEN) {
        # Uses the token only for this push; never written to disk or committed.
        Write-Host "Using GITHUB_TOKEN from environment for non-interactive push."
    } else {
        Write-Host "GITHUB_TOKEN not set - git push may prompt for credentials interactively." -ForegroundColor Yellow
    }

    git push -u origin main
}

Step "Done."
