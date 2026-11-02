# L-CAM

Estúdio desktop de vídeo e webcam construído com o **[Vela Framework](https://github.com/zxlawdx/Vela-framework)**, pywebview, HTML/CSS/JS e Python/OpenCV. O entrypoint e a estrutura gerados por `vela startproject` são preservados.

## Iniciar no Windows (PowerShell)

```powershell
git fetch origin
git switch feat/realtime-video-pipeline
git pull --ff-only origin feat/realtime-video-pipeline
python -m pip install -r requirements.txt
python -m pytest -q
python manage.py collectstatic --no-tailwind --no-install
python manage.py runapp
```

Suporte de CI: **Windows Python 3.12/3.14 e Linux Python 3.12**. A webcam e a GPU específicas do usuário precisam de testes físicos, não reproduzíveis em CI.

## Implementado na branch `feat/realtime-video-pipeline`

- **Workers independentes** para captura e processamento, sem bloquear a UI do Vela. Mailboxes de quadro mais recente (capacidade 1) e gravação em fila limitada; timestamps monotônicos, contadores de perdas e reconexão limitada da câmera.
- **Melhoria clássica** de baixa luminosidade por gamma/CLAHE limitado, suavização cromática, denoising temporal adaptativo ao movimento, controle de brilho/contraste/saturação e nitidez suave. Perfis Originais/Automático/Natural/Baixa luz e comparação original/processada.
- **Prévia pelo Vela**, renderizada apenas quando há novo quadro. Métricas de FPS de captura/processamento/prévia, latência estimada e resolução real.
- **Gravação MP4 independente da prévia**, com controle de timestamps, duplicação/descarte contabilizados e worker próprio.
- **Microfone opcional (experimental)** via `sounddevice`: captura PCM16 com queue limitada e WAV temporário. FFmpeg gera H.264/AAC, alinha início por timestamps e permite verificar streams com ffprobe. Originais são preservados se o mux falhar.
- **ONNX Runtime opcional (experimental)**: manifest local, validação SHA-256, formato BGR↔RGB e NCHW/NHWC, conversão float32, CPU/CUDA/DirectML por sessão e fallback clássico após falha. Carregamento em thread; **não há pesos neurais embarcados**, nem modelos de IA aprovados/testados na GTX 1650.
- **Presets persistentes** com armazenamento atômico e validação local.
- Testes com fontes artificiais sem hardware, CI e diagnósticos para teste de câmera real.

## O que ainda não está concluído

- Benchmark e calibração de 720p/30 FPS na webcam frontal do Acer Nitro, ruído real e qualidade de pele.
- Pelo menos um denoiser e iluminador ONNX **com pesos externos devidamente licenciados** e testados no hardware (o adaptador está pronto, os pesos não estão integrados).
- Validação de sincronização labial e deriva de áudio em gravações longas, NVENC e codecs de fallback.
- Prévia com transporte mais eficiente que JPEG/Base64 pela bridge.
- Webcam virtual, empacotamento/instalador, screenshots reais e teste de interface end-to-end no Windows/Linux.

## Ferramentas e documentação

- [Auditoria forense de 12 repositórios](docs/AUDITORIA_REFERENCIAS_VIDEO_2026.md), com arquivos inspecionados, licenças e riscos.
- [Como usar modelos ONNX locais](docs/COMO_TESTAR_MODELOS_ONNX.md).
- [Gravação com microfone e FFmpeg](docs/GRAVACAO_COM_AUDIO.md).
- [Roadmap de qualidade](docs/ROADMAP_QUALIDADE_VIDEO.md) e [arquitetura/boas práticas](docs/DIRETRIZES_DE_ENGENHARIA.md).

Para medir o hardware real (feche o aplicativo antes, para liberar a câmera):

```powershell
python -m scripts.diagnose_hardware --camera 0 --width 1280 --height 720 --frames 90
```

Para benchmark de inferência com pesos ONNX autorizados:

```powershell
python -m scripts.benchmark_onnx "C:\\caminho\\modelo.json" --provider auto --frames 30
```

Resultados sintéticos de CI **não demonstram FPS real na GTX 1650**.

## Novo experimento — SCUNet Real-PSNR (720p, ONNX)

Agora há uma integração experimental com **SCUNet Real-PSNR**, em vez de depender apenas do DnCNN 320×180. Os pesos ONNX pré-convertidos são baixados no computador do usuário, autenticados por SHA-256 (grafo e tensores externos) e processados em blocos com sobreposição **sem reduzir todo o frame de 1280×720**.

Windows, ambiente `.venv` na pasta do L-CAM:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_scunet_windows.ps1
python manage.py collectstatic --no-tailwind --no-install
python manage.py runapp
```

Na interface Vela: **Modelos ONNX locais → Ativar SCUNet Real-PSNR**. A instalação e o smoke test são executados fora da UI. Não é necessário converter PTH com PyTorch.

A GPU GTX 1650 4 GB não foi testada: o modelo é pesado e pode comprometer o FPS. Veja [documentação e ajustes de memória](docs/SCUNET_REAL_PSNR.md). A saída do modelo e o desempenho são experimentais, **não certificados para gravação profissional**.
