# Gravação MP4 com microfone no L-CAM — fase experimental

## Instalação opcional — Windows

Os filtros clássicos e a gravação sem áudio não exigem dependências extras. Para captura do microfone, instale no ambiente virtual:

```powershell
python -m pip install sounddevice
ffmpeg -version
ffprobe -version
```

O executável FFmpeg e o ffprobe precisam estar disponíveis no `PATH`. Em seguida, reinicie o L-CAM. Selecione um microfone na seção **Gravação** da interface.

## Como funciona

1. `CameraCapture` captura quadros em worker independente, com timestamp `monotonic()`.
2. `RealtimePipeline` aprimora frames e publica resultados para prévia e fila de gravação, sem depender da UI.
3. `RecordingWorker` escreve MP4 temporário com FPS constante, a partir dos timestamps, duplicando frames quando necessário e registrando perdas.
4. `AudioCapture` recebe áudio PCM16 em callback PortAudio, com queue limitada; um worker independente escreve WAV temporário. Nenhuma escrita de disco ocorre dentro do callback.
5. Ao parar, `AsyncFinalizer` executa FFmpeg, alinha os inícios pelo relógio monotônico, gera vídeo H.264/áudio AAC e verifica presença de ambos os streams usando ffprobe, quando disponível.
6. Se FFmpeg falhar, **os temporários .video.mp4 e .audio.wav são preservados**, e a interface informa os caminhos.

### Limitações conhecidas

- O sync inicial depende da precisão dos relógios `monotonic()` nos callbacks do microfone/câmera. Ainda é necessário medir deriva e sincronismo de gravações longas em hardware real.
- O MP4 da câmera sem áudio ainda usa `mp4v` da implementação OpenCV. Com áudio, o FFmpeg tenta `libx264`/AAC na etapa de finalização; se o encoder não estiver instalado a finalização falha e os originais ficam disponíveis.
- O FFmpeg ainda não é distribuído pelo instalador. Instalação externa é necessária.
- O áudio tem buffer limitado; se um callback for perdido por sobrecarga, `dropped_audio_blocks` é registrado. Esses casos exigem melhoria antes de uso profissional.
- Ainda não há monitor visual de níveis de microfone, ajuste de ganho, normalização EBU R128, NVENC, VFR com timestamps contínuos ou garantia de lipsync em longos períodos.

**Validação:** testes unitários simulados verificam WAV e comandos/fracasso de mux. A qualidade do áudio e a sincronização real **não foram testadas com microfone físico**.
