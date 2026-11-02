from vela.core.app import VelaApp

INSTALLED_APPS = ["apps.camera"]

def run():
    app = VelaApp(settings_module="config.settings")
    for app_name in INSTALLED_APPS:
        app.register_app(app_name)
    app.run()
