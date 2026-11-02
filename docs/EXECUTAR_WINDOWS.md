# Executar o L-CAM no Windows (projeto Vela)

O projeto foi organizado segundo o template gerado por `vela startproject`; o app de câmera foi registrado em `config/wsgi.py` como um app Vela normal.

## Setup

No PowerShell, dentro da pasta do repositório:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python manage.py version
python manage.py routes
python manage.py collectstatic --no-tailwind --no-install
python manage.py runapp
```

Também pode ser usado um ambiente virtual existente. A CI passou com Python 3.12 (Windows/Linux) e Python 3.14 (Windows) nas verificações de instalação, testes, rotas e assets. A abertura real do WebView e a webcam ainda precisam ser testadas em hardware.

## Verificações

```powershell
python -m pytest -q
python manage.py doctor --gui qt6
```

## Estrutura

- `manage.py`: launcher oficial do Vela, via `CommandRunner`.
- `config/settings.py`: configurações de aplicação, janela e assets.
- `config/wsgi.py`: registra apps e inicia `VelaApp` com `LCamBridge`.
- `config/bridge.py`: interface explícita Python/JS, herdando a bridge do starter.
- `apps/home/`: starter gerado pelo framework, preservado.
- `apps/camera/`: rota visual, assets e serviços de captura e filtros.
- `staticfiles/`: diretório gerado por `collectstatic`, não versionado.

## Estado funcional e limites

Prévia nativa via OpenCV e controles de brilho/contraste/saturação; gravação MP4 *sem áudio* implementada em nível inicial. O frontend consulta quadros via bridge e ainda não possui pipeline assíncrono otimizado. IA, áudio sincronizado, câmera virtual e executáveis não estão prontos. Câmera e desempenho precisam ser validados no Windows com hardware real.
