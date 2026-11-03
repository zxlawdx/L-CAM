# L-CAM 2026 — investigação forense: webcam de baixa luz, ruído e FPS real

**Estado:** implementação experimental em `feat/lightweight-live-pipeline`, baseada no Vela Framework, OpenCV e Python, sem dependência de novos pesos. **Não** equivale a desempenho/qualidade aprovados no Acer Nitro do usuário: isso precisa ser medido presencialmente.

## 1. Hipótese, evidência e critério de decisão

Hardware observado do usuário: Windows 11, Python 3.14.4, OpenCV 4.14, notebook Acer Nitro GTX 1650 4 GiB; câmera frontal 1280×720, cerca de 10 FPS de captura, filtros antigos 9–10 FPS, custo de filtro mediano de ~70–145 ms e latência ~150–300 ms. O SCUNet Real-PSNR (ONNX Runtime 1.24.4 + DirectML 0) foi consertado com otimizações BASIC e dimensões fixas, mas processou **19,39 segundos por quadro 1280×720** e 4,06 segundos em 640×360. Uma imagem A/B em intensidade 35% mostrou melhora subjetiva pequena e alguma suavização.

Esses dados indicam **dois gargalos distintos**: (1) captura física/driver já ~10 FPS sem IA, possivelmente por exposição prolongada em pouca luz, sem diagnóstico conclusivo da câmera; (2) filtros e redes pesados. Melhorar apenas IA não aumenta o número de quadros que o sensor fornece. O foco é **melhorar sombras/cores sem desfocar textura e sem aumentar bastante custo por quadro**, e medir sensor e processamento separadamente.

Critérios de aceite:

- CPU primeiro, nenhum novo modelo, instalação, ferramenta CUDA ou driver obrigatório.
- Preservar detalhes de luminância e proteger altas luzes já brancas.
- Não manter quadros antigos em filas; não reduzir 720p inteiro a 180p.
- Denoising temporal **somente** em regiões estáticas e **somente** crominância; nada de filtro temporal borrando contorno do rosto.
- Expor **Leve vs Clássico** na interface e permitir comparar exatamente o mesmo quadro.
- Relatório real de média, p95 e captura física, sem alegar 30 FPS de webcam quando não foram medidos.

## 2. Fontes inspecionadas e constatações

### A. IXC Camera — medições REAIS e código (MIT)

- Repo: https://github.com/infernoxc/ixc-camera
- Licença: https://github.com/infernoxc/ixc-camera/blob/main/LICENSE (MIT)
- Relatório medido: https://github.com/infernoxc/ixc-camera/blob/main/docs/performance.md
- Código de processamento: https://github.com/infernoxc/ixc-camera/blob/main/src/processing/image_pipeline.cpp
- Código de denoising temporal: https://github.com/infernoxc/ixc-camera/blob/main/src/processing/temporal_denoise.cpp
- Diagnóstico de exposição: https://github.com/infernoxc/ixc-camera/blob/main/src/camera/exposure_governor.cpp

**Fatos medidos no PC do autor (não no Acer Nitro):** o pipeline C++/SSE2 de cor + nitidez custou ~0,87 ms por quadro 720p; só cor ~0,43 ms. A câmera Lenovo com modos anunciados de 30 FPS forneceu ~19,9 FPS em luz ambiente e passou a 30 com mais luz, por exposição automática. MJPG nativo permitia 720p30, enquanto YUY2 descomprimido chegava a 720p10 na conexão observada. A documentação do projeto declara que o hardware fraco ainda não foi validado. Portanto, esses números **não** são garantia de webcam a 30 FPS nem de nosso Python.

**Técnicas confirmadas no código:** tabela de tons Y e crominância; sharpening com limiar e controle de halo; temporal que limita mistura em movimento; observação do FPS antes de mexer em exposição. O L-CAM implementa seu próprio código em OpenCV Python; nenhuma rotina C++ foi copiada.

### B. BigCam — design OpenCV de baixo custo (GPL-3.0)

- Repo: https://github.com/biglinux/bigcam
- Código lido: https://github.com/biglinux/bigcam/blob/main/usr/share/biglinux/bigcam/core/effects.py
- Licença: https://github.com/biglinux/bigcam/blob/main/COPYING (GPL-3.0)

O código usa `cv2.LUT` para gamma, CLAHE na luminância LAB e `cv2.bilateralFilter` com redução a 50% da imagem quando passa de 480p. A documentação relata otimização de custos, mas **não há benchmark do Acer Nitro com essas funções**. A ideia de fazer processamento caro em menor resolução é válida; a diferença deliberada no L-CAM é **nunca reduzir luminância/contornos do rosto**: apenas suavizar canais cromáticos em resolução reduzida, devolvendo a mistura ao quadro nativo. Nenhum código GPL foi transplantado.

### C. ProMatte — engenharia de fluxo sem atrasar renderização (GPL-2.0-or-later)

- https://github.com/scornik/ProMatte
- https://github.com/scornik/ProMatte/blob/main/docs/performance.md

Usa worker independente, mailbox newest-frame, qualidade adaptativa, refinamento de bordas e estabilização temporal, com testes descritos em documentação. Porém é ferramenta de **segmentação de primeiro plano**, não denoising de webcam e não fornece uma prova de PSNR/qualidade do L-CAM. Nossa arquitetura já tinha capture/process workers e latest-frame mailbox; ela foi preservada, sem copiar componentes licenciados em GPL.

### D. FastLLVE — potencial futuro, mas não primeiro MVP

- https://github.com/Wenhao-Li-777/FastLLVE
- Código do projeto sob Apache-2.0, pesos pré-treinados em links externos na documentação.
- Paper ACM MM 2023: `FastLLVE: Real-Time Low-Light Video Enhancement with Intensity-Aware Lookup Table`.

A proposta LUT aprendível de baixa latência e consistência temporal é relevante, mas modelos/pesos, exportação ONNX e estabilidade na **GTX 1650 + ORT 1.24** não foram medidos. Instalar agora seria repetir o risco operacional do SCUNet. Recomendamos fazer uma prova de desempenho isolada depois do baseline OpenCV, sem prometer qualidade.

### E. NightLift/OpenCV e Beecam — não escolhidos como solução principal

- NightLift: https://github.com/LIUZHIYON/OpenCV-Low-Light-Enhancement usa histogramas, CLAHE, gamma, Retinex e seleção automática para arquivos; não há benchmark confiável de 720p na webcam do usuário. O arquivo `LICENSE` não foi encontrado no local esperado durante a revisão; não incorporar seu código.
- Beecam: https://github.com/tamnguyenvan/beecam reporta ~24 FPS em Intel i5-9500F, mas a tarefa é **segmentar fundo** usando OpenVINO, não remover ruído colorido ou melhorar exposição. Desempenho não é transferível.
- SCUNet: mantido opcional no L-CAM para fotos/pós-processamento, não modo ao vivo.

## 3. Fóruns, relatos práticos e vídeo

- Fórum OBS: https://obsproject.com/forum/threads/getting-the-most-out-of-your-webcam.1036/ — MJPG em resoluções altas e exposição automática podem explicar câmera que apresenta ~10 FPS em baixa luz. É um guia antigo, útil como hipótese, não prova para a webcam deste usuário.
- Relato mais recente OBS: https://obsproject.com/forum/threads/camera-frame-rate-issues.119429/ — luz insuficiente e shutter prolongado causam borramento de movimento e queda de FPS; reduzir tempo de exposição pode aumentar ruído.
- Reddit 2024: https://www.reddit.com/r/Twitch/comments/1eoa1g4/ — várias respostas relatam que luz frontal e configuração de exposição ajudaram; opinião comunitária, não ensaio de hardware controlado.
- Reddit 2026: https://www.reddit.com/r/obs/comments/1uy1hnq/spent_3_weeks_getting_my_cams_dialed_in_obs_so/ — sugestão de bloquear exposição e white balance depois de preparar a iluminação, relato particular.
- Vídeo YouTube (descrição): https://www.youtube.com/watch?v=I2c-jMWmc1o — `How to optimise WEBCAM on OBS…` (GigiGoneMad, 17/04/2026). A descrição sugere resolução/FPS explícitos, exposição/white balance manuais e nitidez moderada. **Foi consultada a descrição/metadata, não medido o vídeo ou reproduzido todo o tutorial.**
- Referência de fabricante: https://help.elgato.com/hc/en-us/articles/4413074299533-Elgato-Facecam-Optimize-Facecam-Image-Quality — aumentar luz recebida pelo sensor e controlar tempo de exposição é mais eficaz do que denoise excessivo; configurações exatas do Facecam **não** se aplicam automaticamente à webcam Acer.

**Conclusão sobre hardware:** nenhum filtro OpenCV ou IA pode reconstruir pixels já saturados de branco ou eliminar motion blur causado por shutter lento. O aplicativo não impõe exposição manual no driver, porque os valores variam entre dispositivos e isso poderia piorar a imagem. Há um diagnóstico somente leitura para observar formato, FPS real e estatísticas de luz.

## 4. Implementação realizada

Arquivos:

- `apps/camera/services/fast_live.py` — `FastLiveEnhancer`: converte BGR→YCrCb uma vez; usa amostra 160×90 para estimar percentil 15 de iluminação (paredes muito claras não ocultam o sujeito escuro); suaviza a estimativa entre quadros e aplica LUT de 256 posições apenas em sombras, protegendo os valores ≥224. Para reduzir granulação cromática, desce crominância à meia resolução, filtra levemente e mistura temporalmente apenas regiões estáticas. Luminância nativa nunca é reduzida ou alisada pelo denoiser; nitidez opcional com limites conservadores. Configurações manuais têm saturação correta (sem `convertScaleAbs` refletir negativos).
- `apps/camera/services/profiled.py` — `ProfiledEnhancer` encapsula novo filtro e `BasicEnhancer` antigo, com reset ao trocar estratégia. Inversão de dependência mantida.
- `apps/camera/services/filters.py` — adiciona `ImageSettings.profile` com valores `fast/legacy`.
- `apps/camera/services/studio.py`, `config/bridge.py` — integração Vela do perfil, persistência com `PresetStore`, fallback após erro neural, sem refatoração destrutiva.
- `apps/camera/templates/index.html`, `apps/camera/static/js/app.js` — seletor de motor visível e explicação honesta, atualização de FPS captura/filtro; IA SCUNet continua separada.
- `apps/camera/services/capture.py` — leitura de backend/FourCC/valores brutos de exposição e ganho, sem modificá-los.
- `scripts/diagnose_exposure.py` — tempo de captura real, FPS, formatos e proporção de áreas escuras/estouradas, relatório local.
- `scripts/benchmark_live.py` — mede fast e legacy no **mesmo** quadro, média/p95, exporta PNG original, fast, legacy e composição A/B, sem alegar FPS total do aplicativo.
- `tests/unit/test_fast_live.py`, `tests/unit/test_capture_exposure.py`, `tests/unit/test_benchmark_live.py` — preservação de bordas, cores, exposição, cena/movimento, retorno à fonte e contrato de perfil.

**Sem nova dependência obrigatória além do OpenCV e NumPy já instalados. Sem download de pesos. Sem alterar os arquivos SCUNet.**

## 5. Benchmarks e evidência: só números honestos

**L-CAM original, notebook Acer Nitro do usuário:** captura ~10 FPS, processamento clássico ~9–10 FPS, filtros frequentemente 70–145 ms. Essas medições foram passadas pelo usuário; iluminação e drivers variam. O SCUNet em 720p fez ~19,39 segundos/quadro, portanto não serve para real-time.

**CI Windows, Python 3.14, runner GitHub Actions** (execução `41557913375`, quadro sintético):

- Filtro Fast 1280×720, 8 frames após warmup: **13,53 ms de média, p95 15,93 ms**.
- Fast vs Legacy 320×180, mesmos pixels, 4 frames: **1,66 ms vs 4,33 ms**, velocidade 2,61×. Esses números vêm do runner Windows; não comparar diretamente com o Acer, nem extrapolar 2,61× a 720p.
- Testes automatizados: 125 aprovados nessa revisão e verificações de rotas/arquivos Vela e JS. CI de mudanças posteriores deve ser consultada no GitHub.

**Teste exploratório em CPU local** (fora do Acer): o protótipo Fast 1280×720 mostrou ~6,3 ms de média neste ambiente. Esse número também não é do usuário, não deve ser anunciado como FPS real da webcam.

**Comparação exploratória com imagem enviada pelo usuário** (metade original de imagem A/B redimensionada): o algoritmo inicial falhou em detectar o rosto escuro porque a parede branca dominava a média. Após mudar a estatística para percentil 15, uma região de pele ficou em torno de Y=96,6→106,2, enquanto uma região da parede ficou Y=253,67→253,68. Não há recuperação da área de clipping e isso **não** prova melhora perceptual em movimento.

## 6. Reproduzir e validar na máquina-alvo

Feche o L-CAM e abra a pasta do projeto no PowerShell com `.venv` ativo:

```powershell
git fetch origin
git switch feat/lightweight-live-pipeline
git pull --ff-only origin feat/lightweight-live-pipeline
python -m pytest -q
# CPU: medir novo motor vs anterior e criar PNG lado a lado na webcam real
python -m scripts.benchmark_live --camera 0 --frames 15 --save
# Captura física do sensor e driver, sem filtros, Vela ou IA
python -m scripts.diagnose_exposure --camera 0 --width 1280 --height 720 --fps 30 --frames 90
# Executar L-CAM
python manage.py collectstatic --no-install
python manage.py runapp
```

Na interface: **02 / Aprimoramento → Motor de imagem → Leve · OpenCV**, modo Automático, intensidade 40%, nitidez 10%, denoise ativado. Alterne para **Clássico** com a webcam já aberta; observe captura FPS, processamento FPS e latência. O teste deve ser repetido sob a mesma iluminação, com movimento de cabeça para identificar fantasmas de denoise, e depois com luz frontal moderada.

Se o usuário quiser usar uma foto original já gravada em vez da câmera:

```powershell
python -m scripts.benchmark_live --input "C:\Users\barro\Pictures\L-CAM\Comparisons\20261102-205134-original.png" --frames 20 --save
```

As imagens ficam em `Pictures/L-CAM/Comparisons`. Os relatórios de diagnóstico físico ficam em `logs/camera-exposure-*.json`.

### Critérios finais de aceite no Acer

1. O perfil leve deve apresentar menor custo por quadro que o clássico, com fotos A/B sem suavização excessiva de cabelo, contornos, óculos e roupa.
2. Ao mudar modos, nenhuma falha na bridge nem no registro de vídeo; histórico temporal é reiniciado.
3. Se captura ficar ~10 FPS mas filtro novo puder acompanhar, o aplicativo mostra orientação de iluminação/exposição em vez de prometer que a IA criará frames.
4. Caso o novo filtro não dê ganho perceptual, preservar opção de legado, analisar qualidade do sensor/iluminação e só então considerar modelo neural pequeno (FastLLVE ou denoiser diferente, com pesos/licenças e benchmarks verificados).

**Status:** implementation e CI não substituem revisão visual e testes de captura/vídeo no notebook do usuário. A branch é experimental e não foi mesclada na `main`.
