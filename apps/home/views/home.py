"""Render the starter through Vela's existing template engine."""

from config.settings import APP_TITLE

from vela import __version__
from vela.template_engine import render_template


def home_view(params: dict) -> str:
    return render_template(
        "apps/home/templates/home.html",
        context={"app_title": APP_TITLE, "vela_version": __version__},
        router=params["router"],
    )
