from vela.template_engine import render_template

def camera_view(params):
    return render_template("apps/camera/templates/index.html", {}, router=params["router"])
