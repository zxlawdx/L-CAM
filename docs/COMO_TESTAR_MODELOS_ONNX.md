# L-CAM: integração experimental de modelos ONNX

O L-CAM inclui uma **infraestrutura de inferência neural opcional**. Não há pesos ONNX empacotados, baixados ou ativados automaticamente. Os modelos DnCNN, SCI, FastDVDnet e Zero-DCE++ são *candidatos investigados*, não modelos que já receberam teste real na GTX 1650.

## Por que o modelo é externo?

Uma rede pré-treinada pode exigir restrições próprias de licença e redistribuição, entradas/tensores específicos ou treinamento para ruído sintético distinto de webcam. **Zero-DCE++** declara uso acadêmico não comercial. **SCI** precisa de verificação de licenciamento. Os pesos de qualquer modelo precisam ter origem, versão, SHA-256 e licença verificados por quem os utilizar.

## Instalação opcional

A infraestrutura principal continua funcionando sem ONNX. Use uma instalação adequada ao sistema e ao Python selecionado:

- Windows CPU: `python -m pip install onnxruntime`
- Windows DirectML: `python -m pip install onnxruntime-directml`
- Windows CUDA: `python -m pip install onnxruntime-gpu` **somente** se a combinação CUDA/cuDNN/driver for compatível.
- Linux CPU: `python -m pip install onnxruntime`

Evite misturar os pacotes de runtime no mesmo ambiente sem verificar as compatibilidades. Consulte a [documentação oficial do DirectML](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html): requer modo sequencial, memória-padrão desabilitada e uma chamada `Run` por vez na mesma sessão.

A GTX 1650 não é uma placa RTX com Tensor Cores; a infraestrutura ONNX pode ser testada na CPU, CUDA ou DirectML quando disponíveis, mas **nenhum FPS específico foi validado**. O SDK NVIDIA Maxine VFX atual exige Tensor Cores, portanto não foi escolhido como dependência.

## Manifest de um modelo local

Crie um JSON com estes campos, na mesma pasta do arquivo .onnx:

```json
{
  "name": "Denoiser ONNX autorizado",
  "task": "denoise",
  "model_path": "meu-modelo.onnx",
  "sha256": "SUBSTITUA_POR_HASH_SHA256_DE_64_CARACTERES",
  "license": "LICENCA_VERIFICADA_DO_MODELO",
  "source": "URL_EXATA_DOS_PESOS",
  "layout": "NCHW",
  "output_kind": "enhanced",
  "input_width": 320,
  "input_height": 180
}
```

Calcule hash no Windows:

```powershell
Get-FileHash .\meu-modelo.onnx -Algorithm SHA256
```

- `task`: `denoise` ou `low_light`.
- `layout`: `NCHW` ou `NHWC`.
- `output_kind`: `enhanced` (rede retorna pixels corrigidos em RGB float32 [0,1]) ou `residual` (rede retorna resíduo a subtrair).
- `input_width`, `input_height`: opcionais juntos. Defina dimensões de entrada suportadas pelo modelo.
- Modelos suportados neste estágio precisam de **uma entrada e uma saída**, ambos tensors 4D RGB float32; redes temporais com cinco frames, mapas de ruído adicionais ou múltiplas saídas **não são compatíveis com este adaptador**.

Na interface desktop, abra **Modelos ONNX locais**, informe caminho completo do manifest, escolha Auto/CUDA/DirectML/CPU e pressione **Carregar modelo**. O aplicativo prepara a sessão em segundo plano e informa provedor e erro real. **Desativar IA** restaura o filtro clássico.

Esta implementação **não garante** que um ONNX arbitrário funcione nem que o modelo rode totalmente na GPU; `get_providers()` descreve o provedor configurado, e kernels individuais podem usar fallback CPU. Meça latência e utilização efetiva.

### Próximas tarefas

Converter e auditar um DnCNN autorizado, checar hash/licença dos pesos, executar vídeo real na GTX 1650, verificar preservação de cor e reconstrução de detalhes. Em uma etapa posterior, implementar múltiplas entradas e janelas temporais para FastDVDnet.
