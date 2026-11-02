"""Entry point seguindo o template gerado por vela startproject."""
import sys

from config import settings
from config.bridge import LCamBridge
from vela.cli.collectstatic import collect_static
from vela.core.app import VelaApp

INSTALLED_APPS = ["apps.home", "apps.camera"]


def run():
    if not getattr(sys, "frozen", False):
        report = collect_static(
            output_dir=settings.STATIC_ROOT,
            skip_tailwind=True,
            asset_config=settings.ASSETS,
            allow_install=False,
            verbose=False,
        )
        if report["errors"]:
            raise RuntimeError("Falha ao coletar os assets locais do L-CAM.")
    app = VelaApp(settings_module="config.settings", bridge_class=LCamBridge)
    for app_name in INSTALLED_APPS:
        app.register_app(app_name)
    app.lifecycle.on_shutdown(app.bridge.studio.close)
    app.run()
