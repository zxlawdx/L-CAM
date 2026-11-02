# L-CAM — Plano aprovado e critérios de entrega

## Objetivo
Aplicativo desktop Windows e Linux Mint baseado no Vela Framework para captura de webcam, prévia em tempo real, processamento de imagem e gravação com áudio. Integração de webcam virtual opcional.

## Ordem de implementação
1. MVP: dispositivos, prévia, controles básicos, gravação MP4 com áudio, erros e métricas de FPS.
2. IA: pipeline plugável, fallback CPU, redução de ruído, correção de luz, comparação original/processado, medição de latência.
3. Webcam virtual e distribuição: adaptadores por SO, integração OBS, builds nativos e documentação.

## Arquitetura
- Vela runtime e pywebview para UI; frontend HTML/CSS/JS sem dependência obrigatória de React.
- OpenCV para captura e filtros; FFmpeg para mux de áudio/vídeo.
- ONNX Runtime para inferência opcional, modelo sujeito a validação de licença, compatibilidade e qualidade.
- Serviços separados: capture, devices, processing, recorder, audio, virtual_camera, configuration.
- Comunicação assíncrona e filas limitadas para não bloquear a UI. Priorizar estabilidade da gravação.

## Diretrizes IHC e design
Referências de padrões de uso: OBS Studio (controle de gravação e fontes), aplicativos de câmera (preview central), editores profissionais (painel lateral contextual). Não copiar identidade visual.
- Hierarquia: prévia dominante; botão Gravar/Pausar/Parar claramente diferenciado; ajustes no painel lateral.
- Estados visíveis: câmera ativa, gravando, tempo, microfone, FPS, modelo ativo, erros.
- Heurísticas de Nielsen: feedback, consistência, controle e liberdade, prevenção de erros, reconhecimento em vez de memorização, estética minimalista.
- Tokens de cor, tipografia, espaçamento e foco; contraste e navegação por teclado.
- Layout responsivo e testes em 1280×720, 1366×768 e 1920×1080.
- Comparação antes/depois com alternância acessível; presets e botão Restaurar padrão.
- Nunca iniciar gravação automaticamente nem perder arquivo silenciosamente.

## Testes e evidências
- Unitários: filtros, limites de parâmetros, presets, transições de estado.
- Integração: vídeo sintético, áudio sintético, gravação e validação via ffprobe.
- Sistema: iniciar, gravar, interromper, reiniciar, lidar com câmera indisponível.
- Performance: FPS, latência, CPU/RAM, GPU quando disponível; testes com IA ligada/desligada.
- UI: screenshots por viewport, inspeção de alinhamento, sobreposição, overflow, foco, mensagens e contraste.
- Build: execução congelada em Windows e Linux em runners nativos.
- Não declarar testes de webcam física, GPU GTX 1650 ou webcam virtual aprovados sem hardware real.

## Critérios de aceite
1. Gravação reproduzível, com áudio sincronizado e caminho do arquivo informado.
2. Filtros desativáveis sem interromper a gravação.
3. Interface sem controles cortados e com estados compreensíveis.
4. Testes automatizados e logs reproduzíveis; falhas documentadas.
5. Build instalável ou limitação técnica documentada.

## Restrições e riscos
- Webcam integrada pode ter limite físico de resolução, ruído e baixa luz.
- 720p/30fps é meta, não promessa.
- Áudio sincronizado e webcam virtual requerem testes específicos por plataforma.
- Vela main pode conter mudanças ainda não publicadas; fixar versão/commit após validação.
- Evitar alterar Vela Framework para atender L-CAM sem necessidade e testes próprios.

## Status
Plano aprovado pelo usuário. Este documento é planejamento, não evidência de implementação.
