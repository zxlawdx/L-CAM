# L-CAM

Estúdio desktop de câmera construído **sobre o [Vela Framework](https://github.com/zxlawdx/Vela-framework)**, com frontend HTML/CSS/JS, bridge pywebview/Vela e processamento de vídeo nativo em Python.

## Primeiros passos — Windows / PowerShell

O projeto preserva a estrutura gerada por `vela startproject` e acrescenta `apps/camera` como app Vela:

```powershell
git switch feat/lcam-initial
git pull origin feat/lcam-initial
python -m pip install -r requirements.txt
python manage.py version
python manage.py routes
python manage.py collectstatic --no-tailwind --no-install
python manage.py runapp
```

A CI verifica **Python 3.12 em Windows/Linux e Python 3.14 em Windows** (instalação, CLI, rotas, assets e testes). A interface gráfica e uma webcam física ainda precisam ser testadas localmente.

## Estrutura

| Caminho | Finalidade |
| --- | --- |
| `manage.py` | Arquivo original do template `vela startproject`, via `CommandRunner` |
| `config/settings.py` | Configuração Vela, janela, assets e rota inicial |
| `config/wsgi.py` | Inicialização `VelaApp` com registro dos apps |
| `config/bridge.py` | `LCamBridge`, interface explícita entre JS e serviços Python |
| `apps/home/` | App padrão criado pelo gerador do Vela |
| `apps/camera/views/` | Rota e template renderizados pelo Vela |
| `apps/camera/static/` | CSS e JS copiados para `staticfiles/camera/` |
| `apps/camera/services/` | Captura, filtros, orquestração e gravação, sem dependência do Vela |
| `tests/` | Testes dos serviços sem hardware real |

## Implementado nesta etapa

- Rota de estúdio com layout escuro e controles responsivos.
- Prévia nativa por OpenCV exibida pela bridge Vela ↔ JavaScript.
- Brilho, contraste, saturação e suavização de ruído.
- Seleção de índice de dispositivo (0, 1, 2).
- Gravação MP4 local (vídeo **sem áudio**) em `~/Videos/L-CAM`.
- Encerramento da captura no ciclo de vida da aplicação.
- Testes unitários e pipeline de validação do scaffold em Windows/Linux.

## Limites conhecidos

O projeto ainda é uma **versão inicial**, não a entrega completa: IA de melhoria de vídeo, sincronização de áudio, câmera virtual, otimizações de desempenho, instalação automática e testes com webcam física permanecem pendentes. A prévia transporta JPEG via bridge; esta etapa privilegia um caminho funcional e testável, não desempenho de produção.

Veja [execução no Windows](docs/EXECUTAR_WINDOWS.md), [plano](docs/PLANO_DE_TRABALHO.md) e [diretrizes de engenharia](docs/DIRETRIZES_DE_ENGENHARIA.md).
