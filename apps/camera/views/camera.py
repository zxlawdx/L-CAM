"""Página de câmera renderizada pelo mecanismo oficial do Vela."""
from vela.template_engine import render_template


def camera_view(params: dict) -> str:
    return render_template(
        "apps/camera/templates/index.html",
        context={"app_title": "L-CAM"},
        router=params["router"],
    )
