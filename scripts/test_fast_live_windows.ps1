# Test lightweight L-CAM processing without reinstalling Python or AI weights.
# From the project root:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_fast_live_windows.ps1 -Camera 0
[CmdletBinding()]
param(
    [int]$Camera = -1,
    [string]$InputImage = "",
    [switch]$SkipTests,
    [switch]$RunApp
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Ambiente virtual nao encontrado: $Python"
}
if ($Camera -lt -1 -or $Camera -gt 10) {
    throw 'A webcam deve ter indice de 0 a 10, ou -1 para ignorar.'
}
if ($Camera -ge 0 -and $InputImage) {
    throw 'Use -Camera OU -InputImage, nao os dois.'
}
Set-Location -LiteralPath $Root
if (-not $SkipTests) {
    Write-Host '[1/4] Testes do L-CAM...' -ForegroundColor Cyan
    & $Python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Um ou mais testes falharam; confira o log acima.' }
} else {
    Write-Host '[1/4] Testes automatizados ignorados (-SkipTests).' -ForegroundColor Yellow
}
Write-Host '[2/4] Benchmark sintetico 720p para criar linha de base...' -ForegroundColor Cyan
& $Python -m scripts.benchmark_live --width 1280 --height 720 --frames 15 --skip-legacy
if ($LASTEXITCODE -ne 0) { throw 'Falha no benchmark sintetico.' }
if ($Camera -ge 0) {
    Write-Host '[3/4] Benchmark e comparacao da webcam real: feche OBS e L-CAM!' -ForegroundColor Cyan
    & $Python -m scripts.benchmark_live --camera $Camera --frames 15 --save
    if ($LASTEXITCODE -ne 0) { throw 'Falha no teste A/B da webcam.' }
    Write-Host '[4/4] Medindo FPS real do sensor sem filtros...' -ForegroundColor Cyan
    & $Python -m scripts.diagnose_exposure --camera $Camera --frames 90
    if ($LASTEXITCODE -ne 0) { throw 'Falha no diagnostico de captura.' }
} elseif ($InputImage) {
    Write-Host '[3/4] Comparacao usando um PNG/JPG original existente...' -ForegroundColor Cyan
    & $Python -m scripts.benchmark_live --input $InputImage --frames 15 --save
    if ($LASTEXITCODE -ne 0) { throw 'Falha comparando a imagem informada.' }
    Write-Host '[4/4] Medicao fisica ignorada (nenhuma webcam solicitada).'
} else {
    Write-Host '[3/4] Comparacao real ignorada; use -Camera 0 ou -InputImage caminho.'
    Write-Host '[4/4] Medicao fisica ignorada; use -Camera 0.'
}
Write-Host ''
Write-Host 'Testes concluídos. A/B: Pictures\L-CAM\Comparisons; diagnostico: logs\camera-exposure-*.json' -ForegroundColor Green
if ($RunApp) {
    Write-Host 'Coletando arquivos estaticos Vela e abrindo aplicativo...' -ForegroundColor Cyan
    & $Python manage.py collectstatic --no-install
    if ($LASTEXITCODE -ne 0) { throw 'Falha em collectstatic.' }
    & $Python manage.py runapp
    if ($LASTEXITCODE -ne 0) { throw 'Vela encerrou com erro.' }
} else {
    Write-Host 'Para abrir o app: python manage.py collectstatic --no-install; python manage.py runapp'
}
