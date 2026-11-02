/* L-CAM: thin Vela UI; all frame processing runs in CameraStudio (Python). */
(() => {
  const root = document.getElementById("lcam-app");
  if (!root || root.dataset.mounted === "true") return;
  root.dataset.mounted = "true";

  const $ = id => root.querySelector("#lcam-" + id);
  let disposed = false;
  let ready = false;
  let active = false;
  let recording = false;
  let finalizing = false;
  let originalPreview = false;
  let previewTimer = null;
  let adjustmentTimer = null;
  let lastMetricsAt = 0;
  let lastSequence = -1;
  let modelPoll = null;
  const modes = {
    auto: "Identifica pouca luz e corrige sombras de forma adaptativa.",
    low_light: "Reforça iluminação de sombras, preservando realces quando possível.",
    balanced: "Aumenta moderadamente o contraste local em ambientes iluminados.",
    off: "Desliga o aprimoramento automático. Ajustes manuais continuam disponíveis."
  };

  function notify(message, isError = false) {
    $("message").textContent = message;
    $("message").classList.toggle("error", isError);
  }
  function status(message, state = "") {
    $("status").textContent = message;
    $("status").className = "lcam-status " + state;
  }
  function setControls() {
    $("start").disabled = !ready || active;
    $("stop").disabled = !active || recording;
    $("record").disabled = !active || recording || finalizing;
    $("finish").disabled = !recording;
    $("compare").disabled = !active;
    $("device").disabled = active;
    $("microphone").disabled = recording || finalizing;
  }
  async function call(method, ...args) {
    const api = window.pywebview?.api;
    if (typeof api?.[method] !== "function") {
      throw new Error("A bridge Python do Vela não está disponível.");
    }
    const result = await api[method](...args);
    if (result?.ok === false) throw new Error(result.error || "Falha no backend.");
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
      refreshAudioDevices();
      refreshPresets();
      refreshModelState();
      refreshScunet();
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
      denoise: $("denoise").checked,
      mode: $("mode").value,
      strength: Number($("strength").value) / 100,
      sharpness: Number($("sharpness").value) / 100
    };
  }
  function labels() {
    for (const name of ["brightness", "contrast", "saturation"]) {
      $(name + "-val").textContent = $(name).value;
    }
    $("strength-val").textContent = $("strength").value + "%";
    $("sharpness-val").textContent = $("sharpness").value + "%";
    $("mode-help").textContent = modes[$("mode").value] || "";
    $("strength").disabled = $("mode").value === "off" && !$("denoise").checked;
  }
  async function syncSettings() {
    labels();
    if (!ready) return;
    try {
      await call("camera_configure", settings());
    } catch (error) {
      if (!disposed) notify(error.message, true);
    }
  }
  function metrics(result) {
    const now = performance.now();
    if (now - lastMetricsAt < 450) return;
    lastMetricsAt = now;
    $("resolution").textContent = result.width + " × " + result.height;
    $("fps").textContent = "Captura " + result.capture_fps + " · Process. " + result.process_fps + " · Prévia " + result.preview_fps + " FPS";
    $("processing").textContent = result.processing_ms + " ms/filtro · " + result.latency_ms + " ms/latência";
  }
  async function preview() {
    if (disposed || !active) return;
    try {
      const result = await call("camera_frame", originalPreview);
      if (disposed || !active) return;
      $("frame").src = result.image;
      $("frame").hidden = false;
      $("empty").hidden = true;
      $("live").hidden = false;
      metrics(result);
    } catch (error) {
      if (disposed || !active) return;
      notify(error.message, true);
      active = false;
      recording = false;
      setControls();
      status("● Erro");
      $("live").hidden = true;
      try { await call("camera_stop"); } catch (_) { /* best effort */ }
      return;
    }
    // Nunca acumular requisições: só pedir novo quadro após resposta da bridge.
    if (!disposed && active) previewTimer = setTimeout(preview, 24);
  }
  async function start() {
    $("start").disabled = true;
    notify("Abrindo câmera nativa...");
    try {
      await call("camera_start", Number($("device").value));
      await call("camera_configure", settings());
      if (disposed) return;
      active = true;
      lastSequence = -1;
      status("● Câmera ativa", "active");
      notify("Prévia iniciada. Use Ver original para comparar sem alterar a gravação.");
      setControls();
      preview();
    } catch (error) {
      if (!disposed) notify(error.message, true);
      active = false;
      setControls();
    }
  }
  async function stop() {
    if (recording) return;
    active = false;
    lastSequence = -1;
    clearTimeout(previewTimer);
    setControls();
    try { await call("camera_stop"); }
    catch (error) { notify(error.message, true); }
    $("frame").hidden = true;
    $("frame").removeAttribute("src");
    $("empty").hidden = false;
    $("live").hidden = true;
    $("resolution").textContent = "— resolução";
    $("fps").textContent = "— FPS";
    $("processing").textContent = "— ms/processamento";
    status("● Desligada");
    if (!disposed) notify("Câmera desligada.");
  }
  async function startRecording() {
    $("record").disabled = true;
    try {
      const device = Number($("microphone").value);
      const result = await call("camera_record_start", device);
      recording = true;
      status("● Gravando", "recording");
      notify("Gravando " + (result.audio ? "vídeo com microfone" : "vídeo sem áudio") + ": " + result.path);
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
      finalizing = !!result.finalizing;
      status(finalizing ? "● Finalizando áudio" : "● Câmera ativa", finalizing ? "recording" : "active");
      notify(finalizing ? "Finalizando MP4 com FFmpeg. Aguarde: " + result.path : "Vídeo salvo: " + result.path + " (sem áudio)");
      setControls();
    } catch (error) {
      notify(error.message, true);
      setControls();
    }
  }
  function compare() {
    originalPreview = !originalPreview;
    $("compare").setAttribute("aria-pressed", String(originalPreview));
    $("compare").textContent = originalPreview ? "Ver aprimorada" : "Ver original";
  }
  function reset() {
    $("mode").value = "auto";
    $("strength").value = "40";
    $("sharpness").value = "10";
    $("brightness").value = "0";
    $("contrast").value = "1";
    $("saturation").value = "1";
    $("denoise").checked = true;
    syncSettings();
    notify("Ajustes restaurados para o modo automático.");
  }

  $("start").addEventListener("click", start);
  $("stop").addEventListener("click", stop);
  $("record").addEventListener("click", startRecording);
  $("finish").addEventListener("click", stopRecording);
  $("compare").addEventListener("click", compare);
  $("reset").addEventListener("click", reset);
  for (const name of ["mode", "strength", "sharpness", "brightness", "contrast", "saturation", "denoise"]) {
    $(name).addEventListener("input", () => {
      labels();
      clearTimeout(adjustmentTimer);
      adjustmentTimer = setTimeout(syncSettings, 140);
    });
  }


  async function refreshAudioDevices() {
    if (!ready || disposed) return;
    try {
      const result = await call("camera_audio_devices");
      const select = $("microphone");
      select.innerHTML = "";
      select.append(new Option("Sem áudio (somente vídeo)", "-1"));
      for (const device of result.devices || []) {
        select.append(new Option(device.name + " (" + device.channels + " canais)", String(device.id)));
      }
      $("audio-help").textContent = result.devices?.length
        ? "Para gravar com microfone, selecione o dispositivo. Exige FFmpeg instalado."
        : "Microfone indisponível. Instale sounddevice para habilitar áudio.";
    } catch (error) {
      $("audio-help").textContent = "Não foi possível detectar microfones: " + error.message;
    }
  }
  async function refreshRecordingStatus() {
    if (!finalizing || disposed || !ready) return;
    try {
      const result = await call("camera_record_status");
      const state = result.finalizer?.state;
      if (state === "done") {
        finalizing = false;
        status("● Câmera ativa", "active");
        notify("MP4 com áudio finalizado: " + result.finalizer.output);
        setControls();
      } else if (state === "error") {
        finalizing = false;
        status("● Erro de finalização");
        notify("FFmpeg falhou; vídeo e WAV temporários preservados. " + result.finalizer.error, true);
        setControls();
      }
    } catch (error) {
      notify("Falha ao consultar finalização: " + error.message, true);
    }
  }

  async function refreshPresets() {
    if (!ready || disposed) return;
    try {
      const result = await call("camera_presets_list");
      const select = $("preset-select");
      const previous = select.value;
      select.innerHTML = "";
      select.append(new Option("Selecionar perfil...", ""));
      for (const name of result.presets || []) {
        select.append(new Option(name, name));
      }
      if ([...select.options].some(option => option.value === previous)) select.value = previous;
    } catch (error) {
      $("preset-status").textContent = error.message;
    }
  }
  function presetMessage(message) {
    $("preset-status").textContent = message;
  }
  async function savePreset() {
    const name = $("preset-name").value.trim();
    if (!name) { presetMessage("Informe um nome para o perfil."); return; }
    try {
      await call("camera_presets_save", name);
      await refreshPresets();
      $("preset-select").value = name;
      presetMessage("Perfil salvo: " + name);
    } catch (error) { presetMessage(error.message); }
  }
  async function applyPreset() {
    const name = $("preset-select").value;
    if (!name) { presetMessage("Escolha um perfil."); return; }
    try {
      const result = await call("camera_presets_apply", name);
      const values = result.settings;
      for (const field of ["brightness", "contrast", "saturation"]) $(field).value = values[field];
      $("mode").value = values.mode;
      $("strength").value = Math.round(values.strength * 100);
      $("sharpness").value = Math.round(values.sharpness * 100);
      $("denoise").checked = !!values.denoise;
      labels();
      presetMessage("Perfil aplicado: " + name);
    } catch (error) { presetMessage(error.message); }
  }
  async function deletePreset() {
    const name = $("preset-select").value;
    if (!name) { presetMessage("Escolha um perfil."); return; }
    try {
      await call("camera_presets_delete", name);
      await refreshPresets();
      presetMessage("Perfil excluído: " + name);
    } catch (error) { presetMessage(error.message); }
  }
  $("preset-save").addEventListener("click", savePreset);
  $("preset-apply").addEventListener("click", applyPreset);
  $("preset-delete").addEventListener("click", deletePreset);
  function showModelState(result) {
    const node = $("model-status");
    const state = result?.state || "off";
    node.dataset.state = state;
    const badge = $("ai-indicator");
    if (badge) {
      badge.textContent = state === "ready" ? "IA ativa: " + result.name
        : state === "loading" ? "IA carregando..."
        : state === "error" ? "IA indisponível" : "IA desligada";
    }
    if (state === "ready") {
      node.textContent = "IA ativa: " + result.name + " · sessão " + result.provider + " · " + result.task + ".";
    } else if (state === "loading") {
      node.textContent = "Verificando hash, licença informada e preparando sessão ONNX...";
    } else if (state === "error") {
      node.textContent = "IA indisponível — filtro clássico ativo. " + (result.error || "Falha desconhecida.");
    } else {
      node.textContent = "IA desligada — processamento clássico ativo.";
    }
  }

  async function refreshScunet() {
    if (!ready || disposed) return;
    try {
      const result = await call("camera_model_catalog");
      const model = (result.models || [])[0];
      const installed = !!model?.installed;
      $("scunet-activate").disabled = !installed;
      $("scunet-status").textContent = installed
        ? "SCUNet Real-PSNR instalado: pronto para testar DirectML/CPU."
        : "SCUNet não instalado. Execute o script setup_scunet_windows.ps1.";
      if (installed && $("model-manifest") && !$("model-manifest").value) {
        $("model-manifest").value = model.manifest;
      }
    } catch (error) {
      $("scunet-status").textContent = "Erro ao verificar modelo: " + error.message;
    }
  }
  async function activateScunet() {
    $("scunet-activate").disabled = true;
    // DnCNN's 320x180 upscaling caused blur. SCUNet instead preserves the
    // camera's full resolution and intentionally disables stacked denoisers.
    $("mode").value = "off";
    $("strength").value = "40";
    $("sharpness").value = "10";
    $("denoise").checked = false;
    labels();
    try {
      await syncSettings();
      const selected = $("model-provider").value;
      showModelState(await call("camera_scunet_load", selected));
      notify("SCUNet: preparando ONNX. Aguarde o indicador IA ativa.");
    } catch (error) {
      showModelState({state: "error", error: error.message});
      notify(error.message, true);
    } finally {
      await refreshScunet();
    }
  }
  $("scunet-activate").addEventListener("click", activateScunet);

  async function refreshModelState() {
    if (!ready || disposed) return;
    try {
      showModelState(await call("camera_model_status"));
    } catch (error) {
      $("model-status").dataset.state = "error";
      $("model-status").textContent = "Falha consultando IA: " + error.message;
    }
  }
  async function loadModel() {
    const manifest = $("model-manifest").value.trim();
    if (!manifest) {
      showModelState({ state: "error", error: "Informe o caminho de um manifesto JSON." });
      return;
    }
    $("model-load").disabled = true;
    try {
      showModelState(await call("camera_model_load", manifest, $("model-provider").value));
    } catch (error) {
      showModelState({ state: "error", error: error.message });
    } finally {
      $("model-load").disabled = false;
    }
  }
  async function unloadModel() {
    try {
      showModelState(await call("camera_model_unload"));
    } catch (error) {
      showModelState({ state: "error", error: error.message });
    }
  }
  $("model-load").addEventListener("click", loadModel);
  $("model-unload").addEventListener("click", unloadModel);
  modelPoll = setInterval(() => { refreshModelState(); refreshRecordingStatus(); }, 2000);
  labels();
  setControls();
  window.addEventListener("pywebviewready", connect);
  window.VelaPage = {
    beforeUnmount() {
      disposed = true;
      active = false;
      clearTimeout(previewTimer);
      clearTimeout(adjustmentTimer);
      clearInterval(modelPoll);
      const api = window.pywebview?.api;
      if (typeof api?.camera_stop === "function") api.camera_stop().catch(() => {});
    }
  };
  connect();
})();
