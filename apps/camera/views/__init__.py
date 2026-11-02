from .camera import camera_view

def register_routes(router):
    router.add("/camera", camera_view, name="camera", title="L-CAM Studio", layout="blank", show_in_sidebar=False)
