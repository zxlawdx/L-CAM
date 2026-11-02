# L-CAM — próximo ciclo de evolução da imagem

Atualizado em 02/11/2026. Este roteiro registra o trabalho autorizado para o aplicativo desktop L-CAM baseado no Vela Framework.

## Contexto e diagnóstico do teste real

Equipamento inicial: notebook Acer Nitro com GPU NVIDIA GTX 1650, webcam frontal integrada e Windows; objetivo futuro inclui Linux Mint. Uso: gravação de conteúdo para Instagram e YouTube, com prévia e resultado em tempo real.

O primeiro teste real mostrou o L-CAM abrindo no Vela, camera ativa e preview funcional. A captura de teste em ambiente escuro mostra ruído cromático abundante, perda de detalhes em sombras e pixels suavizados. Na captura enviada, os controles manuais estavam em brilho -59, contraste 2,4 e saturação 1,2; esses valores reduzem a faixa tonal aproveitável. Parte das limitações decorre da luz e do sensor, não do software.

Objetivo: melhoria mensurável, com cores naturais e latência adequada, SEM alterar indevidamente a identidade ou características do rosto.

## Entrega A — estabilização e qualidade clássica (iniciada neste ciclo)

- [x] Preservar o scaffold oficial do Vela (manage.py / config / views / bridge).
- [x] Negociar 1280×720/30fps com o driver, usando fallback em caso de incompatibilidade; exibir resolução REAL.
- [x] Processar ruído de crominância antes de aumentar luminância.
- [x] Correção adaptativa em sombra (gamma) e CLAHE limitado (não estourar realces).
- [x] Modos Original, Automático, Equilibrado e Baixa luminosidade.
- [x] Intensidade, nitidez suave, ajustes manuais e restaurar padrões.
- [x] Comparação de prévia original/aprimorada sem mudar gravação.
- [x] Medidas de FPS e tempo de processamento na interface.
- [x] Testes unitários sintéticos, teste do contrato de comparação e CI.
- [ ] Validar imagem com o hardware do usuário sob iluminação idêntica.
- [ ] Benchmark de latência em 640×480 e 1280×720 usando câmera real.
- [ ] Ajustar ganho e evitar cintilação temporal com base nas medições.
- [ ] Proteger melhor tons de pele e brancos usando perfis opcionais.

Aceitação: sem crashes, sem distorcer cores exageradamente, sombras mais visíveis em baixa luz, sem amplificação excessiva de ruído e FPS visível. O resultado deve ser comparado com a imagem original NA MESMA iluminação e resolução.

## Entrega B — pipeline para produção em tempo real

- Separar worker de captura do loop de UI/bridge, processamento e codificador; usar fila limitada e estratégia de descarte da prévia para evitar latência crescente.
- Evitar custo de converter cada quadro inteiro para JPEG+base64 pela bridge; investigar endpoint MJPEG local ou shared memory adequada ao Vela, mantendo segurança e ciclo de vida.
- Registrar FPS real capturado, processado, exibido e gravado; medir latência, CPU, GPU e dropped frames.
- Escrever áudio e vídeo em timestamps monotônicos, sincronizar/mux via FFmpeg, adicionar gravação MP4 com microfone selecionável.
- Estudar NVENC (NVIDIA) para gravação e fallback x264; testar arquivos reproduzíveis e áudio sincronizado.
- Histórico básico de presets, configurações persistidas e comparação A/B.

Aceitação pretendida (dependente de driver/sensor): perseguir 24–30fps em 720p com prévia estável, sem crescimento ilimitado de memória, e duração de arquivo correta. Caso necessário, modo leve com resolução reduzida.

## Entrega C — aprimoramento com IA (após baseline)

- Criar `FrameProcessor` protocol e `ProcessorFactory` com backends CPU/OpenCV, ONNX e eventualmente CUDA/DirectML.
- Provar primeiro um modelo leve de redução de ruído/baixa luz ou super-resolução, com licença compatível, integridade de pesos e download opcional explícito.
- Inferência assíncrona, fallback automático se driver/modelo indisponível, medição de FPS/VRAM, batch=1.
- Testar diferentes resoluções, benchmark real na GTX 1650 e compatibilidade de Python/Windows; preservar suporte Linux Mint.
- Evitar restauração facial alucinada por padrão. Recursos faciais, se futuramente oferecidos, devem ser opt-in e visualmente identificados.

Aceitação: ganho visual justificável comparado ao modo clássico, sem alterações de identidade, congelamento de frames ou consumo excessivo.

## Entrega D — integração e distribuição

- Webcam virtual (integração pyvirtualcam/OBS ou backend próprio selecionável).
- Interface final: painel de dispositivos, status de microfone, indicadores de captura/qualidade, atalhos, presets, tratamentos de erro e onboarding.
- Testes end-to-end desktop, screenshots reais Windows e Linux, revisão de IHC (Nielsen), acessibilidade de foco/teclado, instalador via buildapp Vela.
- CI Windows e Linux, testes de regressão, observabilidade e documentação de troubleshooting.

## Arquitetura, SOLID e testes

Vela é infraestrutura de apresentação/bridge. `CameraStudio` orquestra casos de uso. `CameraCapture` e `VideoRecorder` são adaptadores; `BasicEnhancer` é uma estratégia de processamento substituível. `ImageSettings` é objeto de valor imutável e validado. A implementação futura separará workers de tempo real sem levar lógica de frames para as views e o JavaScript.

As classes são testadas com mocks/fakes para não exigir webcam em CI. Testes de desempenho com hardware real não podem ser substituídos por testes sintéticos. Sem alegar que uma webcam escura se torna uma câmera profissional apenas pelo software.
