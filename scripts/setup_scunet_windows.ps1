# SCUNet Real-PSNR setup for L-CAM on Windows 11 (Python 3.14 supported).
# Run from project root:
# powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_scunet_windows.ps1
[CmdletBinding()]
param(
    [ValidateSet(256,320,384,448,512)][int]$TileSize = 384,
    [switch]$SkipBenchmark
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Ambiente virtual nao encontrado em $Python. Ative/crie .venv no projeto."
}
Set-Location -LiteralPath $Root
Write-Host '[1/4] Verificando o ambiente do L-CAM...' -ForegroundColor Cyan
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar requirements.txt.' }

Write-Host '[2/4] Preparando ONNX Runtime DirectML para GTX 1650...' -ForegroundColor Cyan
$RuntimeOK = $false
& $Python -c "import onnxruntime as o; assert 'DmlExecutionProvider' in o.get_available_providers(); print('Provedores ONNX:', o.get_available_providers())" 2>$null
if ($LASTEXITCODE -eq 0) { $RuntimeOK = $true }
if (-not $RuntimeOK) {
    & $Python -m pip install "onnxruntime-directml==1.24.4"
    if ($LASTEXITCODE -ne 0) {
        throw 'Falha instalando ONNX Runtime DirectML. Nenhum peso foi instalado.'
    }
    & $Python -c "import onnxruntime as o; print('Provedores ONNX:', o.get_available_providers()); assert 'DmlExecutionProvider' in o.get_available_providers()"
    if ($LASTEXITCODE -ne 0) { throw 'DirectML nao apareceu entre os provedores ONNX.' }
}

Write-Host '[3/4] Baixando pesos ONNX publicados, verificando os SHA-256...' -ForegroundColor Cyan
& $Python -m scripts.install_scunet --tile-size $TileSize
if ($LASTEXITCODE -ne 0) { throw 'Falha instalando SCUNet. Confira rede, hash e espaco em disco.' }

$Manifest = Join-Path $Root 'models\scunet_real_psnr\scunet_real_psnr.json'
if (-not $SkipBenchmark) {
    Write-Host '[4/4] Diagnosticando ONNX CPU / DirectML adaptadores 0 e 1 (subprocessos isolados)...' -ForegroundColor Cyan
    # The old benchmark swallowed the root cause when a native DirectML DLL
    # terminated Python without a traceback. This wrapper captures exit code,
    # stderr and each adapter's GPU execution in its own child process.
    & $Python -u -m scripts.diagnose_scunet --manifest $Manifest --tile-size 256 --timeout 120
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'A GPU ainda nao passou no teste; leia logs/scunet-diagnostic-*.json. O modelo/arquivos permanecem instalados.'
    }
} else {
    Write-Host '[4/4] Diagnostico omitido (-SkipBenchmark).' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Arquivos SCUNet instalados. So ative IA apos o diagnostico DirectML passar.' -ForegroundColor Green
Write-Host "Manifest instalado: $Manifest"
Write-Host 'Para investigar sem instalar novamente: python -u -m scripts.diagnose_scunet'
Write-Host 'Iniciar L-CAM: python manage.py collectstatic --no-tailwind --no-install; python manage.py runapp'
