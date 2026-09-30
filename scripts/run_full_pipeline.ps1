<# GT-RSSM Full Pipeline Script (Section 8)
   Run from the project root: .\scripts\run_full_pipeline.ps1
   
   This script chains every build step from Section 8 of the build spec.
   It is meant to be run step by step — each step prints a checkpoint
   that should be verified before continuing.
#>

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "`n====== Step 1: Environment Check ======" -ForegroundColor Cyan
python -c "import torch, xgboost, shap, captum, scapy, streamlit, fastapi"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Missing dependencies. Run: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host "OK: All dependencies available." -ForegroundColor Green

Write-Host "`n====== Step 2: Data Presence Check ======" -ForegroundColor Cyan
$dataDirs = @(
    "Data/raw/cic2017/csv",
    "Data/raw/ctu13"
)
foreach ($d in $dataDirs) {
    if (-not (Test-Path $d)) {
        Write-Host "WARNING: $d not found. See Section 2 of build instructions." -ForegroundColor Yellow
    } else {
        Write-Host "OK: $d exists." -ForegroundColor Green
    }
}

Write-Host "`n====== Step 3: Ingestion + Cleaning ======" -ForegroundColor Cyan
Write-Host "Running CIC-IDS2017 cleaning..."
python -m data_prep.clean_cic2017
Write-Host "Running CTU-13 cleaning..."
python -m data_prep.clean_ctu13

Write-Host "`n====== Step 4: PCAP Streaming (Wednesday only) ======" -ForegroundColor Cyan
if (Test-Path "Data/raw/cic2017/pcap/Wednesday-workingHours.pcap") {
    Write-Host "Parsing Wednesday PCAP (this may take a while)..."
    python -m src.ingestion.pcap_parser
} else {
    Write-Host "SKIP: Wednesday PCAP not found (optional)." -ForegroundColor Yellow
}

Write-Host "`n====== Step 5: Windowing + Graph Construction ======" -ForegroundColor Cyan
Write-Host "Building CIC-IDS2017 windows..."
python -m src.features.build_dataset --dataset cic2017
Write-Host "Building CTU-13 windows..."
python -m src.features.build_dataset --dataset ctu13
Write-Host "Building CTU-13 background pool..."
python -m src.features.build_dataset --dataset ctu13 --background --skip-splits

Write-Host "`n====== Step 6: Logistic Regression Baseline ======" -ForegroundColor Cyan
python -m src.baseline.logreg_baseline
Write-Host "CHECKPOINT: Check Data/processed/eval_results.md for baseline numbers." -ForegroundColor Yellow

Write-Host "`n====== Step 7: Stage-1 Pretraining ======" -ForegroundColor Cyan
python -m src.training.pretrain_stage1
Write-Host "CHECKPOINT: Check checkpoints/stage1_curves.png — recon should decrease, KL should not collapse." -ForegroundColor Yellow

Write-Host "`n====== Step 8: Stage-2 Fine-tuning ======" -ForegroundColor Cyan
python -m src.training.finetune_stage2
Write-Host "CHECKPOINT: Val infiltration F1 should exceed the LR baseline." -ForegroundColor Yellow

Write-Host "`n====== Step 11: Full Evaluation ======" -ForegroundColor Cyan
python -m src.evaluation.run_eval
Write-Host "CHECKPOINT: Check Data/processed/eval_results.md for the full comparison table." -ForegroundColor Yellow

Write-Host "`n====== Step 12: Streamlit Dashboard ======" -ForegroundColor Cyan
Write-Host "Launch with: streamlit run dashboard/streamlit_app.py" -ForegroundColor Green

Write-Host "`n====== Step 13: FastAPI (optional) ======" -ForegroundColor Cyan
Write-Host "Launch with: uvicorn api.main:app --port 8000" -ForegroundColor Green

Write-Host "`n====== Pipeline Complete ======" -ForegroundColor Green
Write-Host "All build steps completed. Review checkpoints above before proceeding to docs/demo."
