# SCUNet Real-PSNR — qualidade nativa, execução experimental no L-CAM

## Fonte, licença e hashes

Utiliza os arquivos ONNX publicados em https://huggingface.co/Heliosoph/scunet-onnx (Apache-2.0 declarado pelo publicador), derivados de https://github.com/cszn/SCUNet. Confira licenças/atribuição antes de redistribuir.

O modelo consiste em dois arquivos que DEVEM estar juntos:

- scunet_color_real_psnr.onnx — 3,8 MB; SHA-256 231be201ab413dbc999d7951caa9844846b93a12a40a41e037d6b5888ed4e88c
- scunet_color_real_psnr.onnx.data — 73,1 MB; SHA-256 98825ea1210b641c71e5f052f582c70c49fd44b35387ebe2c034268c17df3feb

O instalador baixa e verifica ambos. Nenhum PyTorch, conversão PTH ou pesos em Git são necessários.

## Testar no seu Acer Nitro (Windows / Python 3.14)

Feche a câmera/L-CAM antes. Na pasta do projeto:

    git fetch origin
    git switch feat/scunet-real-denoise
    git pull --ff-only origin feat/scunet-real-denoise
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_scunet_windows.ps1

Depois inicie o L-CAM:

    python manage.py collectstatic --no-tailwind --no-install
    python manage.py runapp

No painel **Modelos ONNX locais**, clique **Ativar SCUNet Real-PSNR**. Aguarde o indicador superior mostrar que a IA está ativa. Se houver erro, informe a mensagem completa.

O botão aplica configurações iniciais de **40%** de intensidade, nitidez 10%, modo clássico OFF e redução de ruído clássico OFF. Isso evita empilhar dois denoisers. Ajuste intensidade para 20–40% conforme o resultado real. Use **Ver original** para comparar.

## Resolução e velocidade

O SCUNet exige entrada RGB float32 NCHW com dimensões múltiplas de oito. O L-CAM usa **blocos sobrepostos de 384 × 384 px** com pesos graduais nas junções e mantém o quadro original 1280 × 720. Não reduz o rosto para 320 × 180 como no teste DnCNN.

Na GTX 1650 (4 GB), esse modelo pode ser **muito lento**. Se a sessão DirectML ficar sem memória, reinstale o manifesto usando:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_scunet_windows.ps1 -TileSize 256

Menor tile reduz VRAM, mas aumenta a quantidade de inferências. Para medir o runtime:

    python -m scripts.benchmark_scunet --provider directml --width 640 --height 360 --frames 3
    python -m scripts.benchmark_scunet --provider directml --width 1280 --height 720 --frames 1

Esse benchmark não mede webcam, renderização do Vela ou gravação. **Não há FPS garantido na GTX 1650.** A CI usa pesos simulados para testes unitários e só comprova integração de contrato.

## Quando treinar seu próprio modelo?

Só depois de medir a velocidade e comparar a preservação de cabelo, pele, roupas, ruído e cintilação em quadros originais sob a mesma iluminação. Para denoising supervisionado, é necessário criar referências limpas confiáveis; treinar usando a saída de outro modelo como verdade absoluta pode preservar os mesmos defeitos e borrar mais.

## Engenharia

- apps/camera/services/inference.py: contratos ONNX, autenticação do arquivo externo, blocos e mesclagem
- apps/camera/services/model_manager.py: estratégia neural, controles e fallback
- apps/camera/services/studio.py e config/bridge.py: catálogo instalado, interface do Vela
- scripts/install_scunet.py: download autenticado/idempotente
- scripts/setup_scunet_windows.ps1: instalador de um comando
- scripts/benchmark_scunet.py: benchmark sintético
- tests/unit/test_scunet.py: validação de imagem, arquivos e segurança sem downloads de CI

Limitações: transporte JPEG/Base64 pela bridge, latência real desconhecida, SCUNet processa quadros individualmente (não vídeo temporal), qualidade final depende da iluminação e sensor. Não são inventados detalhes faciais.

## Comparação de qualidade sem travar a gravação

Se o SCUNet não atingir FPS suficiente no Acer Nitro, use-o para **aprimorar uma fotografia isolada** e comparar a fidelidade. Feche o aplicativo para liberar a webcam e execute:

    python -m scripts.compare_scunet --camera 0 --provider directml --strength 0.4

Ou, para testar um MP4 já gravado, sem precisar ligar a webcam:

    python -m scripts.compare_scunet --input "C:\\Users\\barro\\Videos\\L-CAM\\seu-video.mp4" --provider directml --strength 0.4

Os PNGs ficam em Pictures/L-CAM/Comparisons: imagem original, SCUNet e montagem lado a lado. O original é salvo sem alterações. A inferência pode levar muitos segundos; isso não é um teste de FPS em streaming.

O primeiro teste de inferência **real** com os pesos ONNX (GitHub Actions Linux, CPU) passou em 02/11/2026. Para 256×256 em CPU de CI: aquecimento 1,84 s e quadro posterior 1,76 s. Esses números não representam a GTX 1650 e justificam a comparação por foto antes de prometer tempo real.
