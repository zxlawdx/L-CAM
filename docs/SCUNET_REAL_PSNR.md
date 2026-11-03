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

## Quando o instalador informa "benchmark GPU falhou"

Esse aviso **não** significa que o download falhou: os hashes dos arquivos podem estar íntegros e o provedor DmlExecutionProvider disponível, mas a inferência real pode falhar por drivers, incompatibilidades de operadores, seleção errada da GPU ou falta de VRAM.

**Não reexecute todo o instalador para investigar.** Após atualizar a branch, com a webcam fechada, rode:

    python -u -m scripts.diagnose_scunet

O diagnóstico verifica o manifesto, executa **CPU, DirectML adaptador 0 e DirectML adaptador 1 em processos independentes**, usando blocos 256×256, e registra erro/retorno nativo mesmo que o ONNX Runtime derrube um subprocesso. O relatório fica em `logs/scunet-diagnostic-YYYYMMDD-HHMMSS.json`. Envie a saída do terminal para identificar a causa.

Para investigar um adaptador isoladamente:

    python -u -X faulthandler -m scripts.benchmark_scunet --provider directml --dml-device-id 1 --tile-size 256 --width 256 --height 256 --frames 1

Se o adaptador 1 passar e o 0 falhar, selecione **DirectML — adaptador 1** no painel Vela antes de ativar o SCUNet. O índice não é necessariamente o mesmo listado pelo nvidia-smi; notebooks com GPU integrada e NVIDIA dedicada podem apresentar adaptadores diferentes. O L-CAM agora mostra o índice escolhido no status da IA.

**Não tente forçar o SCUNet na interface antes de uma inferência DirectML passar no diagnóstico**: uma falha nativa do ONNX Runtime no processo principal pode encerrar todo o aplicativo. Se CPU passar e DirectML falhar, a rede ONNX e os pesos são utilizáveis, mas a rota GPU precisa de correção; SCUNet na CPU ainda será provavelmente lento demais para prévia em tempo real.

A documentação do DirectML exige memória-pattern desligada e modo sequencial, que o adaptador do L-CAM configura. A mera presença de `DmlExecutionProvider` não valida execução do modelo.

## Correção do encerramento do L-CAM ao ativar SCUNet

Anteriormente a sessão ONNX Runtime DirectML era criada e executada **dentro do mesmo Python do Vela/pywebview**. Um erro em código nativo pode encerrar o processo sem traceback nem chance de `try/except` funcionar.

Agora `AsyncModelManager` usa `IsolatedOnnxRuntimeEngine`: um **subprocesso Python persistente** carrega a sessão ONNX e recebe os frames via protocolo binário limitado; o processo principal do L-CAM apenas se comunica com ele. Se houver falha nativa, timeout ou frame inválido, `RealtimePipeline` volta ao `BasicEnhancer`, mantém a webcam e gravação abertas e publica a causa no status do modelo. O worker tem encerramento explícito e cada frame é limitado por tamanho e tempo para impedir filas ilimitadas.

Testes de regressão simulam `os._exit()` durante carga e durante um frame, falha de retorno, saída malformada e worker travado. O teste com SCUNet real na CI usa CPU e processo separado; **isso não demonstra compatibilidade com DirectML na GTX 1650**.

Para descobrir a causa de travamento do DirectML sem colocar o aplicativo em risco, feche a câmera e execute:

```powershell
git pull --ff-only origin feat/scunet-real-denoise
python -u -m scripts.diagnose_scunet
```

O diagnóstico testa CPU e adaptadores DirectML 0/1 em subprocessos individuais e salva o relatório JSON em `logs/scunet-diagnostic-*.json`. A disponibilidade de `DmlExecutionProvider` isoladamente não comprova que o modelo executará.

Após diagnosticar, rode `python manage.py collectstatic --no-tailwind --no-install` e `python manage.py runapp`. Se a ativação da IA falhar, ela deverá voltar ao processamento clássico e apresentar erro no painel, em vez de encerrar o Vela. **Não há garantia de que uma falha de driver de vídeo afete apenas o subprocesso**; mantenha drivers e Windows atualizados e reporte o código de saída.

## Correção de encerramento do L-CAM ao ativar SCUNet

A versão inicial criava a sessão ONNX/DirectML no **mesmo processo do Vela**. Um erro nativo do driver podia encerrar o aplicativo sem traceback. A versão atual executa todo o ONNX Runtime em `apps/camera/services/onnx_child.py`, iniciado por `IsolatedOnnxRuntimeEngine` em um processo separado.

- O modelo é carregado e **um tile de inferência real é executado** no filho **antes** de aparecer `IA ativa`.
- Se o subprocesso falhar durante a inicialização ou processamento, o Vela continua ativo, recebe um erro legível e retorna automaticamente aos filtros clássicos.
- O protocolo `onnx_wire.py` transporta apenas JSON e bytes BGR, com limites de tamanho; nenhum pickle.
- Há prazo de 65 s para ativação e limite de 8 s para cada quadro ao vivo. Se SCUNet for lento demais para a GTX 1650, ele é desativado com aviso, sem bloquear indefinidamente a prévia.
- A escolha explícita `DirectML — adaptador 0` ou `adaptador 1` **não pode ser disfarçada como GPU** quando a sessão efetiva é CPU.
- Testes com subprocessos falsos exercitam encerramento nativo, timeout e fallback; CI também executa o **modelo real em um subprocesso CPU**.

Atualize com `git pull --ff-only origin feat/scunet-real-denoise`. **Não precisa baixar os pesos novamente.** Para investigar um notebook com gráficos híbridos (integrada + NVIDIA), feche a câmera e rode:

```powershell
python -u -m scripts.diagnose_scunet
```

O relatório estará em `logs/scunet-diagnostic-*.json`, com CPU e DirectML device_id 0/1. Se nenhum adaptador DirectML passar, **não tente forçar a IA no aplicativo**; envie o relatório para identificar o erro. Na interface, só selecione o adaptador que passou. O índice DXGI não corresponde necessariamente ao número mostrado pelo `nvidia-smi`.

Ainda não há prova de 720p/30FPS na GTX 1650 e uma falha grave de driver em nível de sistema pode afetar a GPU compartilhada, mesmo com processo isolado. Em caso de lentidão, a comparação por foto é o caminho recomendado.

## Diagnóstico conclusivo após o crash `0xC0000005` nos adaptadores 0 e 1

O diagnóstico do Acer Nitro com Python 3.14.4 mostrou que o SCUNet inicia na CPU e realiza inferência (aproximadamente **0,45 FPS para 256×256**), mas **`DmlExecutionProvider` encerra o processo nativo no `session.run()` tanto no adaptador 0 quanto no 1**. Isso não é erro de download, manifesto nem início do Vela.

Para saber se DirectML falha com *qualquer* modelo ou apenas com a arquitetura SCUNet, foi incluído `scripts/diagnose_directml.py`. Ele gera automaticamente dois pequenos ONNX (`Add` e `Conv+Relu`), sem redes externas nem pesos, e executa CPU/DirectML nos dois adaptadores em **subprocessos independentes**. Usa o profiling do ONNX Runtime para exigir ao menos um nó efetivamente executado no DirectML (sem considerar fallback CPU como sucesso).

```powershell
git pull --ff-only origin feat/scunet-real-denoise
python -m pip install "onnx>=1.16,<2"
python -u -m scripts.diagnose_directml
```

O relatório fica em `logs/directml-minimal-*.json`. **Não exige abrir L-CAM, conectar webcam, reinstalar FFmpeg ou baixar o SCUNet novamente.**

Interpretação:

- `Add` e `Conv` falham com `0xC0000005` também: problema provável no runtime/driver DirectML da máquina; primeiro registrar versão do ORT e drivers, depois avaliar teste controlado com outra versão do pacote e, em seguida, CUDA ou WinML.
- `Add` e `Conv` funcionam de verdade no DirectML: suspeita maior de incompatibilidade do SCUNet, sua exportação ONNX, shapes dinâmicos/operadores, otimização de grafo ou recursos de memória. Evitar repetir SCUNet na interface até corrigir esse caso.
- Um adaptador passa e outro falha: selecionar apenas o que executou nós no DirectML, validando também a identidade física da GPU.

**Nota de segurança:** o isolamento do ONNX protege contra encerramentos nativos do processo da aplicação, mas não garante que problemas de driver ou reinicializações da GPU não afetem outros programas. Não repetir testes contínuos quando houver instabilidade do vídeo no Windows.

## DirectML passou nos modelos mínimos; falta discriminar o grafo SCUNet

O teste do usuário com **ONNX Runtime DirectML 1.24.4**, Python 3.14 e Windows 11 confirmou que ambos os adaptadores (0 e 1) executam, de verdade, os operadores `Add` e `Conv+Relu` na GPU. CPU e GPU passaram, e o profiling reportou `gpu_nodes >= 1`, sem fallback CPU nesses modelos.

O SCUNet continua falhando em `session.run()` com `0xC0000005` nos dois adaptadores. Inspeção do arquivo ONNX oficial em CI: **opset 18, 2122 nós**, contendo `Einsum`, `LayerNormalization`, `ScatterND`, `ScatterElements`, `Range` e atenção. Assim, um teste de convolução isolada não é suficiente para provar compatibilidade de toda a rede.

Foi criado `scripts/diagnose_scunet_variants.py` para distinguir problemas de forma dinâmica, otimização de grafo e tamanho de entrada. É um teste **experimental, que executa a GPU** e pode produzir falhas de driver nativas em subprocessos. Feche o L-CAM e outros aplicativos que usam a webcam. Execute, sem reinstalar nada:

```powershell
git pull --ff-only origin feat/scunet-real-denoise
python -u -m scripts.diagnose_scunet_variants
```

O script primeiro testa a forma 64×64 na CPU; depois executa as duas GPUs com as otimizações ONNX desativadas e, se necessário, apenas as básicas. Se alguma configuração passar na GPU, aumenta gradualmente o tamanho até 256×256; falhando, encerra os testes para aquele adaptador. O resultado fica em `logs/scunet-compatibility-*.json`.

Opcional: `python -m scripts.diagnose_scunet_variants --inspect-only` lista os operadores sem iniciar DirectML; `--full` inclui níveis de otimização Extended e All (mais tentativas, execute apenas se necessário).

Não ative SCUNet no streaming ainda. Se a CPU 64×64 falhar, não tire conclusões sobre GPU antes de investigar o formato da entrada. Se uma combinação de GPU passar, será necessário suportá-la explicitamente no carregador e medi-la com blocos reais — não se deve presumir que a sessão padrão do L-CAM herdará as opções do diagnóstico.

Há pacotes `onnxruntime-gpu` para Python 3.14/Windows, mas eles **não devem ser instalados por cima de `onnxruntime-directml` no mesmo ambiente**. Em caso de incompatibilidade persistente, considerar subprocesso CUDA em venv independente, após verificar driver e dependências CUDA/cuDNN.
