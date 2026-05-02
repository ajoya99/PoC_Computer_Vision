(function () {
  var selectedFile = null;
  var showLabels = true;
  var showConfidence = true;
  var CLASS_NAMES = ["box", "pallet", "person", "forklift", "cart", "wheelchair"];
  var MODEL_OPTIONS = [
    { key: "rtdetr", label: "RT-DETR-L" },
    { key: "yolo",   label: "YOLO26s" },
  ];
  var currentModelIdx = 0;

  // ---------------------------------------------------------------------------
  // Server URL — persisted in localStorage, configurable from the UI
  // ---------------------------------------------------------------------------
  var SERVER_URL_KEY = "cv-server-url";
  var DEFAULT_SERVER_URL = "http://127.0.0.1:8765";

  function getServerUrl() {
    return (localStorage.getItem(SERVER_URL_KEY) || DEFAULT_SERVER_URL).replace(/\/$/, "");
  }

  function saveServerUrl(url) {
    var clean = (url || "").trim().replace(/\/$/, "");
    if (!clean) clean = DEFAULT_SERVER_URL;
    localStorage.setItem(SERVER_URL_KEY, clean);
    return clean;
  }

  // ---------------------------------------------------------------------------
  // Server status dot
  // ---------------------------------------------------------------------------
  function setServerDot(state) {
    var dot = document.getElementById("serverStatusDot");
    if (!dot) return;
    dot.className = "server-status-dot " + (state || "");
  }

  async function pingServer(url) {
    try {
      await fetch(url + "/detect", {
        method: "OPTIONS",
        headers: { "ngrok-skip-browser-warning": "1" },
      });
      setServerDot("online");
    } catch (e) {
      setServerDot("offline");
    }
  }

  function initServerConfig() {
    var input = document.getElementById("serverUrlInput");
    var btn   = document.getElementById("serverSaveBtn");
    if (!input || !btn) return;

    var saved = getServerUrl();
    input.value = saved;
    pingServer(saved);

    btn.addEventListener("click", function () {
      var url = saveServerUrl(input.value);
      input.value = url;
      setServerDot("");
      pingServer(url);
    });

    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") btn.click();
    });
  }

  // ---------------------------------------------------------------------------

  var dropZone;
  var imageInput;
  var previewImage;
  var emptyState;
  var overlayInfo;
  var runBtn;
  var clearBtn;
  var labelsToggle;
  var confidenceToggle;
  var modelToggle;
  var metricImage;
  var metricStatus;
  var metricDetections;

  function updateMetric(el, value) {
    if (el) el.textContent = value;
  }
  function setStatus(status) { updateMetric(metricStatus, status); }
  function setDetectionCount(v) { updateMetric(metricDetections, v); }
  function setImageMetric(name) { updateMetric(metricImage, name || "Not loaded"); }

  function getClassConf() {
    var map = {};
    CLASS_NAMES.forEach(function (cls) {
      var el = document.getElementById("conf-" + cls);
      map[cls] = el ? parseFloat(el.value) : 0.25;
    });
    return map;
  }

  function initSliders() {
    CLASS_NAMES.forEach(function (cls) {
      var slider = document.getElementById("conf-" + cls);
      var valEl  = document.getElementById("val-" + cls);
      if (!slider) return;

      function updateFill() {
        var min = parseFloat(slider.min);
        var max = parseFloat(slider.max);
        var val = parseFloat(slider.value);
        var pct = ((val - min) / (max - min) * 100).toFixed(1) + "%";
        slider.style.setProperty("--pct", pct);
        if (valEl) valEl.textContent = val.toFixed(2);
      }

      slider.addEventListener("input", updateFill);
      updateFill();
    });
  }

  function renderPreview(file) {
    if (!file) {
      selectedFile = null;
      previewImage.removeAttribute("src");
      previewImage.style.display = "none";
      emptyState.classList.remove("hidden");
      overlayInfo.classList.add("hidden");
      setImageMetric("Not loaded");
      setStatus("Waiting");
      setDetectionCount("-");
      return;
    }

    selectedFile = file;
    var reader = new FileReader();
    reader.onload = function (event) {
      previewImage.src = event.target.result;
      previewImage.style.display = "block";
      emptyState.classList.add("hidden");
      overlayInfo.classList.remove("hidden");
      overlayInfo.textContent = "Detection output will appear after running the model.";
      setImageMetric(file.name);
      setStatus("Ready");
      setDetectionCount("-");
    };
    reader.readAsDataURL(file);
  }

  function fileFromDropEvent(event) {
    var files = event.dataTransfer && event.dataTransfer.files;
    if (!files || files.length === 0) return null;
    return files[0];
  }

  function onImageChosen(event) {
    var files = event.target.files;
    if (files && files.length > 0) renderPreview(files[0]);
  }

  function setToggleState(button, enabled) {
    button.setAttribute("aria-pressed", enabled ? "true" : "false");
    button.textContent = enabled ? "ON" : "OFF";
  }

  async function runDetection() {
    if (!selectedFile) {
      setStatus("No image");
      overlayInfo.classList.remove("hidden");
      overlayInfo.textContent = "Select or drag an image first.";
      return;
    }

    setStatus("Running...");
    setDetectionCount("...");
    overlayInfo.classList.remove("hidden");
    overlayInfo.textContent = "Sending image to inference server...";
    runBtn.disabled = true;

    try {
      var imageB64 = await new Promise(function (resolve, reject) {
        var reader = new FileReader();
        reader.onload = function (e) { resolve(e.target.result); };
        reader.onerror = reject;
        reader.readAsDataURL(selectedFile);
      });

      var classConf = getClassConf();
      var minConf   = Math.min.apply(null, Object.values(classConf));
      var modelKey  = MODEL_OPTIONS[currentModelIdx].key;
      var serverUrl = getServerUrl();

      var response = await fetch(serverUrl + "/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "1" },
        body: JSON.stringify({
          image_b64:  imageB64,
          confidence: minConf,
          class_conf: classConf,
          model_key:  modelKey,
        }),
      });

      if (!response.ok) {
        var errText = await response.text();
        throw new Error("Server error " + response.status + ": " + errText);
      }

      var result = await response.json();

      if (result.annotated_image) {
        previewImage.src = result.annotated_image;
        previewImage.style.display = "block";
        emptyState.classList.add("hidden");
      }

      var count = typeof result.detections === "number" ? result.detections : 0;
      setDetectionCount(String(count));
      setStatus("Done");
      setServerDot("online");

      var summary = "";
      if (result.items && result.items.length > 0) {
        var counts = {};
        result.items.forEach(function (d) {
          counts[d.class_name] = (counts[d.class_name] || 0) + 1;
        });
        summary = Object.keys(counts).map(function (k) {
          return counts[k] + "\u00d7\u00a0" + k;
        }).join("  \u00b7  ");
      } else {
        summary = "No objects detected above the set thresholds.";
      }
      overlayInfo.textContent = summary;

    } catch (error) {
      var msg = error && error.message ? error.message : String(error);
      setServerDot("offline");
      if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("Load failed")) {
        setStatus("Server offline");
        overlayInfo.textContent = "Cannot reach server. Check the URL above or start the server + ngrok.";
      } else {
        setStatus("Error");
        overlayInfo.textContent = "Detection failed: " + msg;
      }
      setDetectionCount("-");
    } finally {
      runBtn.disabled = false;
    }
  }

  function initDragAndDrop() {
    ["dragenter", "dragover"].forEach(function (eventName) {
      dropZone.addEventListener(eventName, function (event) {
        event.preventDefault();
        dropZone.classList.add("drag-over");
      });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
      dropZone.addEventListener(eventName, function (event) {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
      });
    });

    dropZone.addEventListener("drop", function (event) {
      var file = fileFromDropEvent(event);
      if (file) renderPreview(file);
    });

    dropZone.addEventListener("click", function () {
      imageInput.click();
    });
  }

  function init() {
    dropZone    = document.getElementById("dropZone");
    imageInput  = document.getElementById("imageInput");
    previewImage = document.getElementById("previewImage");
    emptyState  = document.getElementById("emptyState");
    overlayInfo = document.getElementById("overlayInfo");
    runBtn      = document.getElementById("runBtn");
    clearBtn    = document.getElementById("clearBtn");
    labelsToggle      = document.getElementById("labelsToggle");
    confidenceToggle  = document.getElementById("confidenceToggle");
    metricImage       = document.getElementById("metricImage");
    metricStatus      = document.getElementById("metricStatus");
    metricDetections  = document.getElementById("metricDetections");
    modelToggle       = document.getElementById("modelToggle");

    if (modelToggle) {
      modelToggle.addEventListener("click", function () {
        currentModelIdx = (currentModelIdx + 1) % MODEL_OPTIONS.length;
        var opt = MODEL_OPTIONS[currentModelIdx];
        modelToggle.textContent = opt.label;
        modelToggle.dataset.modelKey = opt.key;
      });
      modelToggle.textContent = MODEL_OPTIONS[0].label;
    }

    initServerConfig();
    initSliders();
    initDragAndDrop();

    imageInput.addEventListener("change", onImageChosen);
    runBtn.addEventListener("click", runDetection);

    clearBtn.addEventListener("click", function () {
      imageInput.value = "";
      renderPreview(null);
    });

    labelsToggle.addEventListener("click", function () {
      showLabels = !showLabels;
      setToggleState(labelsToggle, showLabels);
    });

    confidenceToggle.addEventListener("click", function () {
      showConfidence = !showConfidence;
      setToggleState(confidenceToggle, showConfidence);
    });

    setToggleState(labelsToggle, showLabels);
    setToggleState(confidenceToggle, showConfidence);
    renderPreview(null);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
