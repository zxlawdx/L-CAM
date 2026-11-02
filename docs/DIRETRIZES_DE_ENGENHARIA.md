# Diretrizes de engenharia — L-CAM

Este documento complementa docs/PLANO_DE_TRABALHO.md e é requisito de implementação.

## POO e SOLID
- Responsabilidade única: separar captura, dispositivos, filtros, inferência, gravação, áudio, webcam virtual e UI.
- Extensibilidade por composição e protocolos tipados; não modificar o núcleo para adicionar novo filtro.
- Substituição: implementações devem respeitar contratos de frames, lifecycle, erros e recursos.
- Interfaces segregadas: CameraSource, FrameProcessor, RecordingSink, AudioSource, VirtualCameraSink.
- Inversão de dependências: casos de uso dependem de protocolos, nunca de OpenCV, FFmpeg, Vela ou ONNX diretamente.

## Clean Architecture
- domain/: entidades, value objects, estados e protocolos, sem dependências externas.
- application/: casos de uso (start_preview, start_recording, stop_recording, update_effects).
- infrastructure/: adaptadores OpenCV, FFmpeg, ONNX, dispositivos, armazenamento e virtual camera.
- presentation/: endpoints/bridge Vela e frontend HTML/CSS/JS.
- Composition root centraliza configuração e criação das dependências.

## Padrões de projeto — apenas quando justificáveis
- Strategy: processadores de frames substituíveis.
- Adapter: integração de câmeras, áudio, FFmpeg, ONNX e webcam virtual.
- Factory: seleção de implementações por disponibilidade do hardware.
- State: estados Idle, Previewing, Recording, Stopping, Error, com transições explícitas.
- Observer/Event Bus: notificações de estado e métricas para UI.
- Dependency Injection: facilitar mocks e testes sem hardware.
Evitar classes artificiais, abstrações prematuras e hierarquias excessivas.

## Qualidade
- Python com type hints, dataclasses quando adequadas, logging estruturado e exceptions específicas.
- Ruff e pytest em CI, testes de contratos e de transição de estados.
- Sem lógica de negócio em views, endpoints ou callbacks JS.
- Gerenciamento explícito de threads, buffers, backpressure e liberação de câmera/arquivo.
- Cobertura focada em regras críticas, não apenas porcentagem.
- Código documentado em português ou inglês de modo consistente.
- Avaliar performance antes de otimizar; IA deve ter fallback e desativação simples.

## Critérios de revisão
Toda PR deve explicar arquitetura, testes executados, riscos e evidências de interface quando houver alterações visuais.
