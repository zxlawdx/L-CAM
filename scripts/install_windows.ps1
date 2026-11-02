# L-CAM Windows: dependencias de video, modelo DnCNN e verificacao local.
# Executar na raiz: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_windows.ps1
[CmdletBinding()]
param(
    [switch]$SkipModel,
    [switch]$SkipFFmpeg
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Run-Python([string[]]$Arguments) {
    & $script:Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python falhou (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}
function Add-UserPath([string]$Directory) {
    $env:Path = "$Directory;$env:Path"
    $currentUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $currentUserPath) { $currentUserPath = '' }
    $known = @($currentUserPath -split ';' | Where-Object { $_ })
    if ($known -notcontains $Directory) {
        [Environment]::SetEnvironmentVariable('Path', (($known + $Directory) -join ';'), 'User')
        Write-Host "Adicionado ao PATH do usuario: $Directory"
    }
}
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Ambiente virtual nao encontrado em $Python. Crie .venv primeiro."
}
Set-Location -LiteralPath $Root
Write-Host '[1/5] Preparando dependencias no ambiente do L-CAM...' -ForegroundColor Cyan
Run-Python -Arguments @('-m','pip','install','-r',(Join-Path $Root 'requirements.txt'))
Run-Python -Arguments @('-m','pip','install','onnxruntime-directml==1.24.4','sounddevice>=0.5,<1')
Run-Python -Arguments @('-c', "import onnxruntime as ort; print('ONNX providers:', ort.get_available_providers())")

if (-not $SkipFFmpeg) {
    Write-Host '[2/5] Baixando FFmpeg 9.0.2 / FFprobe (SHA-256 fixado)...' -ForegroundColor Cyan
    $Bin = Join-Path $Root 'tools\ffmpeg'
    $Ffmpeg = Join-Path $Bin 'ffmpeg.exe'
    $Ffprobe = Join-Path $Bin 'ffprobe.exe'
    if (-not ((Test-Path -LiteralPath $Ffmpeg) -and (Test-Path -LiteralPath $Ffprobe))) {
        $Zip = Join-Path $env:TEMP 'lcam-ffmpeg-9.0.2-essentials.zip'
        $Unpack = Join-Path $env:TEMP 'lcam-ffmpeg-9.0.2-unpack'
        $Url = 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip'
        # https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip.sha256
        $Expected = '60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba'
        if (-not (Test-Path -LiteralPath $Zip) -or
            (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) {
            Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
        }
        $Actual = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Expected) {
            throw "FFmpeg: SHA-256 incorreto. Esperado: $Expected; recebido: $Actual"
        }
        if (Test-Path $Unpack) { Remove-Item $Unpack -Force -Recurse }
        Expand-Archive -LiteralPath $Zip -DestinationPath $Unpack
        $Executable = Get-ChildItem -Path $Unpack -Filter 'ffmpeg.exe' -File -Recurse | Select-Object -First 1
        if (-not $Executable) { throw 'ZIP sem ffmpeg.exe.' }
        $ProbeSource = Join-Path $Executable.Directory.FullName 'ffprobe.exe'
        if (-not (Test-Path -LiteralPath $ProbeSource)) { throw 'ZIP sem ffprobe.exe.' }
        New-Item -Path $Bin -ItemType Directory -Force | Out-Null
        Copy-Item -LiteralPath $Executable.FullName -Destination $Ffmpeg -Force
        Copy-Item -LiteralPath $ProbeSource -Destination $Ffprobe -Force
        Remove-Item -LiteralPath $Unpack -Force -Recurse
    }
    Add-UserPath $Bin
    & $Ffmpeg -version | Select-Object -First 1
    & $Ffprobe -version | Select-Object -First 1
}

if (-not $SkipModel) {
    Write-Host '[3/5] Instalando PyTorch CPU apenas para converter peso...' -ForegroundColor Cyan
    Run-Python -Arguments @('-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch==2.10.0')
    Run-Python -Arguments @('-m','pip','install','onnx==1.23.0')
    Write-Host '[4/5] Baixando peso DnCNN do autor e convertendo para ONNX...' -ForegroundColor Cyan
    Run-Python -Arguments @('-m','scripts.prepare_dncnn')
    $manifest = Join-Path $Root 'models\dncnn_color_blind\dncnn_color_blind.json'
    Write-Host '[5/5] Testando inferencia DirectML...' -ForegroundColor Cyan
    & $Python -m scripts.benchmark_onnx $manifest --provider directml --frames 10
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Benchmark DirectML falhou; nao considerar GPU validada. Teste CPU com --provider cpu.'
    }
    Write-Host "Na interface L-CAM, carregue o manifest: $manifest"
} else {
    Write-Host 'Modelo omitido por -SkipModel.'
}
Write-Host 'Instalacao encerrada. Abra outro PowerShell para carregar PATH persistente.' -ForegroundColor Green
Write-Host 'Para iniciar o L-CAM: python manage.py runapp'
