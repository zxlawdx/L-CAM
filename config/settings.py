"""Configuração do L-CAM baseada no template oficial do Vela."""

APP_TITLE = "L-CAM"
APP_VERSION = "0.1.0"
APP_ICON = ""
APP_AUTHOR = "L-CAM contributors"
APP_DESCRIPTION = "Estúdio de webcam nativo construído sobre Vela Framework."
APP_FILE_EXTENSIONS = []

ENTRY_ROUTE = "/camera"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

LAYOUT = {"sidebar": False, "topbar": False, "theme": "dark"}

API = {
    "enabled": True,
    "host": "127.0.0.1",
    "port": 8000,
    "auto_port": True,
    "server": "waitress",
    "workers": 4,
    "docs_enabled": False,
}
DEBUG = True
STATIC_ROOT = "staticfiles"
ASSETS = {"tailwind": False}
