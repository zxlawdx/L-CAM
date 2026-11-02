/* L-CAM frontend: device I/O is owned by CameraStudio through the Vela bridge. */
(() => {
  const root = document.getElementById("lcam-app");
  if (!root || root.dataset.mounted === "true") return;
  root.dataset.mounted = "true";

  const $ = id => root.querySelector("#lcam-" + id);
  let disposed = false, active = false, recording = false, ready = false;
  let previewTimer = null, adjustmentTimer = null;

  function notify(message, isError = false) {
    $("message").textContent = message;
    $("message").classList.toggle("error", isError);
  }
  function status(text, state = "") {
    $("status").textContent = text;
    $("status").className = "lcam-status " + state;
  }
  function setControls() {
    $("start").disabled = !ready || active;
    $("stop").disabled = !active || recording;
    $("record").disabled = !active || recording;
    $("finish").disabled = !recording;
    $("device").disabled = active;
  }

  async function call(method, ...args) {
    const api = window.pywebview?.api;
    if (typeof api?.[method] !== "function") {
      throw new Error("A bridge Python do Vela não está disponível.");
    }
    const result = await api[method](...args);
    if (result && result.ok === false) throw new Error(result.error || "Falha no backend.");
    return result;
  }

  async function connect() {
    try {
      const pong = await call("ping");
      if (pong !== "pong") throw new Error("Resposta inesperada da bridge.");
      if (disposed) return;
      ready = true;
      $("runtime").textContent = "● Vela conectado";
      setControls();
    } catch (error) {
      if (disposed) return;
      ready = false;
      $("runtime").textContent = "● Python indisponível";
      notify(error.message, true);
      setControls();
    }
  }

  function settings() {
    return {
      brightness: Number($("brightness").value),
      contrast: Number($("contrast").value),
      saturation: Number($("saturation").value),
      denoise: $("denoise").checked
    };
  }
  async function syncSettings() {
    for (const name of ["brightness", "contrast", "saturation"]) {
      $(""+name+"-val").textContent = $(""+name).value;
    }
    if (!ready) return;
    try { await call("camera_configure", settings()); }
    catch (error) { notify(error.message, true); }
  }

  async function preview() {
    if (!active || disposed) return;
    try {
      const result = await call("camera_frame");
      if (!active || disposed) return;
      $("frame").src = result.image;
      $("frame").hidden = false;
      $("empty").hidden = true;
      $("live").hidden = false;
    } catch (error) {
      if (!active || disposed) return;
      notify(error.message, true);
      active = false;
      recording = false;
      setControls();
      status("● Erro", "");
      $("live").hidden = true;
      return;
    }
    if (active && !disposed) previewTimer = setTimeout(preview, 90);
  }

  async function start() {
    $("start").disabled = true;
    notify("Abrindo câmera nativa...");
    try {
      await call("camera_start", Number($("device").value));
      await call("camera_configure", settings());
      if (disposed) return;
      active = true;
      status("● Câmera ativa", "active");
      notify("Prévia nativa iniciada. Filtros processados por OpenCV.");
      setControls();
      preview();
    } catch (error) {
      notify(error.message, true);
      active = false;
      setControls();
    }
  }

  async function stop() {
    if (recording) return;
    active = false;
    clearTimeout(previewTimer);
    setControls();
    try { await call("camera_stop"); }
    catch (error) { notify(error.message, true); }
    $("frame").hidden = true;
    $("frame").removeAttribute("src");
    $("empty").hidden = false;
    $("live").hidden = true;
    status("● Desligada");
    if (!disposed) notify("Câmera desligada.");
  }

  async function startRecording() {
    $("record").disabled = true;
    try {
      const result = await call("camera_record_start");
      recording = true;
      status("● Gravando", "recording");
      notify("Gravando MP4 sem áudio em: " + result.path);
      setControls();
    } catch (error) {
      notify(error.message, true);
      setControls();
    }
  }

  async function stopRecording() {
    $("finish").disabled = true;
    try {
      const result = await call("camera_record_stop");
      recording = false;
      status("● Câmera ativa", "active");
      notify("Vídeo salvo em: " + result.path + " (sem áudio)");
      setControls();
    } catch (error) {
      notify(error.message, true);
      setControls();
    }
  }

  $("start").addEventListener("click", start);
  $("stop").addEventListener("click", stop);
  $("record").addEventListener("click", startRecording);
  $("finish").addEventListener("click", stopRecording);

  for (const name of ["brightness", "contrast", "saturation", "denoise"]) {
    $(name).addEventListener("input", () => {
      for (const field of ["brightness", "contrast", "saturation"]) {
        $(field + "-val").textContent = $(field).value;
      }
      clearTimeout(adjustmentTimer);
      adjustmentTimer = setTimeout(syncSettings, 130);
    });
  }

  window.addEventListener("pywebviewready", connect);
  window.VelaPage = {
    beforeUnmount() {
      disposed = true;
      active = false;
      clearTimeout(previewTimer);
      clearTimeout(adjustmentTimer);
      const api = window.pywebview?.api;
      if (typeof api?.camera_stop === "function") api.camera_stop().catch(() => {});
    }
  };
  connect();
})();
