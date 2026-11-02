# L-CAM 2026 — relatório de engenharia (02/11/2026)

Branch: `feat/realtime-video-pipeline`. Origem: `feat/lcam-initial`; o scaffold oficial do Vela permanece intacto. Este relatório diferencia implementação, validação automatizada e **pendências em hardware real**.

## P0 — investigação e auditoria

- Auditoria de **12 repositórios** em [AUDITORIA_REFERENCIAS_VIDEO_2026.md](AUDITORIA_REFERENCIAS_VIDEO_2026.md).
- Foram consultados os arquivos de implementação pedidos para FastDVDnet, Zero-DCE++, SCI, BGOBS, npuscale, IXC Camera, NVIDIA VFX SDK, KAIR, Real-ESRGAN, RVM e Beecam. Para o LiveVisionKit arquivado só foi inspecionado o README, explicitamente sinalizado.
- Licenças verificadas por metadados públicos: MIT (FastDVDnet, KAIR, npuscale, IXC), GPL-3.0 (BGOBS, RVM, LiveVisionKit), BSD-3-Clause (Real-ESRGAN), Apache-2.0 (Beecam). Zero-DCE++ **não comercial** no README; SCI não tem SPDX disponível pelo GitHub. Esses dados **não substituem verificação de pesos**.
- Nenhum algoritmo de fonte externa foi copiado; os módulos L-CAM são implementações próprias que seguem estratégias arquiteturais observadas.

## P1 — captura desacoplada e gravação

Estado: **implementação funcional no código, validação automatizada sem hardware**.

Fluxo:

```text
Device/CameraCapture
       │ (monotonic timestamp)
       ▼
CaptureWorker → LatestFrameMailbox[1] → ProcessorWorker
                                            │
                                    ProcessedFrame
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                      PreviewMailbox[1]           RecordingQueue[max 90]
                             │                             │
                    Vela Bridge / JPEG             RecordingWorker / MP4
                             │                             │
                          pywebview                WAV + FFmpeg/AAC (opcional)
```

- `mailbox.py`: objetos `CapturedFrame` e `ProcessedFrame`, mailbox condition newest-only.
- `realtime.py`: threads separadas e contadores `captured_frames`, `processed_frames`, `dropped_preprocessing`, `capture_fps`, `process_fps`, `reconnects`.
- `recording_worker.py`: queue finita, codificador isolado, ritmo de quadros por captura monotônica, duplicações/descartes contados.
- `studio.py`: orquestração; prévia não lê webcam nem executa filtros.
- `config/bridge.py`: mantém APIs oficiais do Vela e expõe somente operações validadas.
- `apps/camera/static/js/app.js`: solicita apenas quadros novos por sequência, sem criar requests simultâneas.

Pendência técnica: transporte JPEG/Base64 ainda consume CPU e largura de memória; para baixar latência avaliar MJPEG via servidor local Vela ou outro transporte validado. Encoder OpenCV produz vídeo CFR: a fidelidade dos timestamps é aproximada por duplicação e descarte; não é ainda um recorder VFR profissional.

## P2 — redução de ruído e iluminação

Estado: **implementado o fallback clássico, com testes sintéticos**.

- `filters.py`: chroma denoise, correção adaptativa por gamma, CLAHE fraco, limite de exposição, contraste/saturação manuais e unsharp moderado.
- `temporal.py`: mistura adaptativa com máscara por diferença de movimento em luminância, reset por corte de cena, sem lookahead.
- `BasicEnhancer.reset()`: chamado ao iniciar/reconectar a câmera e alternar modo.
- Proteção da resposta à exposição por média suavizada, reduzindo flicker.
- Limitante: ainda não há balanço de branco especializado, medição de ruído físico por modelo de câmera, nitidez mascarada por textura e testes em cenas reais com rosto em movimento.

## P3 — sistema neural opcional

Estado: **adaptador ONNX e seleção de provedor concluídos; modelos pré-treinados reais NÃO integrados**.

- `inference.py`: manifesto local obrigatório com SHA-256/licença/fonte, limitação de tamanho, contrato de tensor float32 RGB `NCHW`/`NHWC` com 3 canais, normalização e pós-processamento.
- `model_manager.py`: carregamento assíncrono, estratégia intercambiável, retomada de processamento clássico se falhar a inferência.
- Seleção do provedor existente CUDA, DirectML ou CPU. `DmlExecutionProvider` habilita execução sequencial e desabilita memória-padrão; a sessão é protegida contra `Run` concorrente.
- Nenhum pacote ONNX é instalado por padrão; nenhum peso de licença desconhecida é baixado.
- **Ainda falta** escolher modelos DnCNN/iluminação licenciados, converter quando necessário e validar pesos, compatibilidade de formato, latência, saída visual e GPU real. Modelos temporais multi-input (FastDVDnet) permanecem futuros.

Documentação: [COMO_TESTAR_MODELOS_ONNX.md](COMO_TESTAR_MODELOS_ONNX.md).

## P4 — áudio e gravação

Estado: **implementação experimental, validação sintética; sem evidência de sincronização real**.

- `audio.py`: captura do microfone no backend PortAudio/sounddevice, fila limitada e WAV temporário.
- `ffmpeg_mux.py`: finalização em thread, H.264/AAC, offset pelo início monotônico, validação com ffprobe e `os.replace`. Arquivos brutos são preservados se o mux falhar.
- `studio.py`: permite gravar sem áudio por padrão ou ativar microfone selecionado, somente quando sounddevice/FFmpeg estiverem presentes.
- `apps/camera/templates/index.html`: interface para microfone e status de finalização.
- **Ainda falta** validar com microfone real, gravações longas, deriva, áudio com timestamp por amostra, câmera desconectada durante escrita, encoder NVENC e retenção/disponibilidade de ffmpeg empacotado.

Documentação: [GRAVACAO_COM_AUDIO.md](GRAVACAO_COM_AUDIO.md).

## P5 — interface, testes e validação

Estado: **CI automatizada, interface a validar presencialmente**.

- UI mantém identidade visual azul-escuro do L-CAM, modos com comparação original, parâmetros explícitos, controles de IA em painel recolhido, perfis salvos, microfone e métricas.
- `presets.py`: repositório local de perfis imutáveis, validação, limite de 30 e escrita atômica.
- Testes unitários e com fakes: formatos ONNX, validação de hash, captura/preview independentes, desacoplamento de gravação, buffers limitados, anti-ghosting, reconexão parcial, WAV/mux, perfis, UI e modelos com fallback.
- GitHub Actions roda no Windows Python 3.12/3.14 e Linux 3.12, compilação Python, pytest, `node --check`, `routes`, `collectstatic` e benchmark sintético.

**Sem validação presencial:** qualidade em iluminação baixa com rosto em movimento, comparação original/aprimorado no mesmo quadro, 720p/30fps no Acer Nitro, GTX 1650 ONNX real, seleção de microfone, sincronização/lipsync, snapshots reais de desktop/UX, memória em testes longos e instalador via Vela.

### Medir o hardware real

```powershell
python -m scripts.diagnose_hardware --camera 0 --width 1280 --height 720 --frames 90
```

O script registra hardware reportado, provedores ONNX disponíveis, FPS sequencial de captura+filtro, p95 de captura/processamento e resolução **real** retornada pelo driver. Fecha webcam no `finally`.

Para modelo local e hash verificados:

```powershell
python -m scripts.benchmark_onnx "C:\\Modelos\\modelo.json" --provider auto --frames 20
```

Não transformar esses resultados em FPS da UI inteira: a primeira medição não inclui pywebview/bridge, e a segunda não inclui webcam/codec.

## Arquivos criados ou alterados

- **Novos**: `services/mailbox.py`, `services/realtime.py`, `services/recording_worker.py`, `services/temporal.py`, `services/inference.py`, `services/model_manager.py`, `services/audio.py`, `services/ffmpeg_mux.py`, `services/presets.py`, `scripts/diagnose_hardware.py`, `scripts/benchmark_onnx.py` e testes específicos.
- **Modificados**: `services/studio.py`, `services/filters.py`, `config/bridge.py`, `apps/camera/templates/index.html`, `apps/camera/static/js/app.js`, `README.md` e CI.
- **Preservados**: `manage.py` oficial do gerador, `config/wsgi.py` e a infraestrutura do Vela; nenhum arquivo de `Vela-framework` foi alterado.

## Riscos e ordem recomendada a seguir

1. Reproduzir teste real de webcam+gravação em ambiente iluminado e escuro; coletar diagnóstico e screenshots com exatamente a mesma luz.
2. Medir captura 720p e localizar gargalos CPU (filtro temporal, JPEG/Base64, encoder). Se a prévia não acompanhar, reduzir estágio mais pesado e manter janela newest-only.
3. Escolher/validar **um** modelo DnCNN com pesos licenciados; testar CPU/DirectML/CUDA e comparar fidelidade de rostos. Depois um modelo leve de iluminação com licença apropriada.
4. Ensaiar áudio real longo, verificador de A/V via ffprobe e testar recuperação após falhas.
5. Substituir o transporte de prévia e completar câmera virtual/instaladores/UX após dados reais.

**Nota:** a implementação não transforma ruído de sensor em detalhes que não foram capturados; qualquer restauração facial de IA pode introduzir características inexistentes e deve ficar opcional.
