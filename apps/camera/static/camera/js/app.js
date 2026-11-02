(() => {
  const $ = id => document.getElementById(id);
  let stream = null, recorder = null, chunks = [];
  const say = msg => { $("message").textContent = msg; };
  async function listDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === "videoinput");
    $("device").innerHTML = '<option value="">Padrão do sistema</option>';
    for (const cam of cameras) {
      const option = document.createElement("option");
      option.value = cam.deviceId; option.textContent = cam.label || "Câmera";
      $("device").append(option);
    }
  }
  async function start() {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw Error("WebView sem suporte a getUserMedia. Use o adaptador nativo de captura.");
      const id = $("device").value;
      stream = await navigator.mediaDevices.getUserMedia({ video: id ? {deviceId:{exact:id}} : true, audio: true });
      $("video").srcObject = stream; $("placeholder").hidden = true;
      $("status").textContent = "Câmera ativa"; $("start").disabled = true; $("stop").disabled = false; $("record").disabled = false;
      await listDevices(); say("Câmera iniciada.");
    } catch(e) { say("Erro ao abrir câmera/microfone: " + e.message); }
  }
  function stop() {
    if(recorder?.state === "recording") return;
    stream?.getTracks().forEach(t => t.stop()); stream = null; $("video").srcObject = null;
    $("placeholder").hidden = false; $("status").textContent = "Câmera desligada";
    $("start").disabled = false; $("stop").disabled = true; $("record").disabled = true; say("Câmera desligada.");
  }
  function record() {
    try {
      if (!window.MediaRecorder) throw Error("MediaRecorder indisponível neste WebView.");
      chunks = []; recorder = new MediaRecorder(stream);
      recorder.ondataavailable = e => { if(e.data.size) chunks.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunks, {type: recorder.mimeType || "video/webm"});
        const url = URL.createObjectURL(blob), a = document.createElement("a");
        a.href = url; a.download = "lcam-recording.webm"; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        say("Gravação finalizada. Confira o arquivo salvo.");
      };
      recorder.start(1000); $("record").disabled = true; $("finish").disabled = false; $("stop").disabled = true; $("status").textContent = "● Gravando";
    } catch(e) { say("Erro na gravação: " + e.message); }
  }
  function finish() {
    if(recorder?.state === "recording") recorder.stop();
    $("record").disabled = false; $("finish").disabled = true; $("stop").disabled = false; $("status").textContent = "Câmera ativa";
  }
  $("start").addEventListener("click", start); $("stop").addEventListener("click", stop);
  $("record").addEventListener("click", record); $("finish").addEventListener("click", finish);
  listDevices().catch(() => {});
  window.addEventListener("beforeunload", () => stream?.getTracks().forEach(t => t.stop()));
})();