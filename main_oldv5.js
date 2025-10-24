console.log("🚀 LockedIn: Gaze Keyboard + Live Cursor + Fixed Calibration");

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const textContainer = document.getElementById("textContainer");
const keys = document.querySelectorAll(".key");
const gazeCursor = document.getElementById("gaze-cursor");

const MODEL_SIZE = 384;
const DWELL_TIME = 2000;
let session = null;
let modelLoaded = false;
let cameraReady = false;
let dwellTimer = null;
let activeKey = null;
let currentHoveredKey = null;
let calibrated = false;

// Calibration elements
const calibrationOverlay = document.getElementById("calibration-overlay");
const calibrationInstruction = document.getElementById("calibration-instruction");
const thumbsUp = document.getElementById("thumbs-up");

// === Initialize Webcam ===
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await new Promise((resolve) => {
      video.onplaying = () => {
        cameraReady = true;
        console.log("✅ Webcam live");
        resolve();
      };
    });
  } catch (err) {
    console.error("❌ Camera access failed:", err);
  }
}

// === Load Model ===
async function loadModel() {
  try {
    console.log("📦 Loading model...");
    session = await ort.InferenceSession.create("./models/best.onnx");
    modelLoaded = true;
    console.log("✅ Model loaded");
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
}

// === Manual Typing (for debugging or fallback) ===
keys.forEach((key) => {
  key.addEventListener("click", () => {
    const val = key.textContent.trim();
    if (key.classList.contains("delete")) {
      textContainer.textContent = textContainer.textContent.slice(0, -1);
    } else if (key.classList.contains("space")) {
      textContainer.textContent += " ";
    } else {
      textContainer.textContent += val;
    }

    key.classList.add("highlight");
    setTimeout(() => key.classList.remove("highlight"), 300);
  });
});

// === Calibration Logic ===
function startCalibration() {
  calibrationOverlay.style.display = "flex";
  calibrationInstruction.textContent =
    "Click the 'G' key and look at it for 2 seconds to calibrate";

  const gKey = Array.from(keys).find(
    (k) => k.textContent.trim().toLowerCase() === "g"
  );

  if (gKey) {
    gKey.addEventListener(
      "click",
      async () => {
        calibrationInstruction.textContent = "Calibrating... keep looking 👀";

        // === Move the cursor to the G key center (absolute positioning) ===
        const rect = gKey.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        gazeCursor.style.transform = `translate(${centerX}px, ${centerY}px)`;
        gazeCursor.style.display = "block";
        gazeCursor.style.opacity = "1"; // fade in

        // Highlight feedback
        gKey.classList.add("highlight");
        setTimeout(() => gKey.classList.remove("highlight"), 800);

        // Simulate dwell for 2 seconds
        await new Promise((resolve) => setTimeout(resolve, 2000));

        // ✅ Success visual feedback
        thumbsUp.style.display = "block";
        calibrationInstruction.textContent = "Calibration successful!";

        setTimeout(() => {
          thumbsUp.style.display = "none";
          calibrationOverlay.style.display = "none";
          calibrated = true;
          detect(); // start inference loop
        }, 1000);
      },
      { once: true }
    );
  }
}

// === Model Inference Loop ===
async function detect() {
  if (!modelLoaded || !cameraReady || !calibrated)
    return requestAnimationFrame(detect);

  const offscreen = document.createElement("canvas");
  offscreen.width = MODEL_SIZE;
  offscreen.height = MODEL_SIZE;
  const offctx = offscreen.getContext("2d");
  offctx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const img = offctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const data = new Float32Array(MODEL_SIZE * MODEL_SIZE * 3);
  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    data[i] = img.data[i * 4] / 255;
    data[i + MODEL_SIZE * MODEL_SIZE] = img.data[i * 4 + 1] / 255;
    data[i + 2 * MODEL_SIZE * MODEL_SIZE] = img.data[i * 4 + 2] / 255;
  }

  const tensor = new ort.Tensor("float32", data, [1, 3, MODEL_SIZE, MODEL_SIZE]);

  try {
    const result = await session.run({ [session.inputNames[0]]: tensor });
    const output = result[session.outputNames[0]];
    drawGaze(output);
  } catch (err) {
    console.error("❌ Inference error:", err);
  }

  requestAnimationFrame(detect);
}

// === Draw gaze points and handle key hover ===
function drawGaze(output) {
  let detections = [];
  const width = canvas.width;
  const height = canvas.height;

  if (output.data) {
    const data = output.data;
    const stride = output.dims?.[2] || 6;
    for (let i = 0; i < data.length; i += stride) {
      detections.push(data.slice(i, i + stride));
    }
  }

  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(video, 0, 0, width, height);

  let sumX = 0,
    sumY = 0,
    valid = 0;
  detections.forEach(([x, y, w, h, conf]) => {
    if (conf < 0.4) return;
    const xC = (x / MODEL_SIZE) * width;
    const yC = (y / MODEL_SIZE) * height;
    ctx.beginPath();
    ctx.arc(xC, yC, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "lime";
    ctx.fill();
    sumX += xC;
    sumY += yC;
    valid++;
  });

  if (valid > 0) {
    const gx = sumX / valid;
    const gy = sumY / valid;

    ctx.beginPath();
    ctx.arc(gx, gy, 7, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();
    handleGazeSelection(gx, gy);
  }
}

// === Gaze to Keyboard Mapping ===
function handleGazeSelection(gx, gy) {
  const previewRect = canvas.getBoundingClientRect();

  // Map camera gaze → screen coordinates
  const pageX = previewRect.left + (gx / canvas.width) * previewRect.width;
  const pageY = previewRect.top + (gy / canvas.height) * previewRect.height;

  // Move the red cursor
  gazeCursor.style.transform = `translate(${pageX}px, ${pageY}px)`;

  // Find which key the gaze is over
  const key = Array.from(keys).find((k) => {
    const r = k.getBoundingClientRect();
    return pageX >= r.left && pageX <= r.right && pageY >= r.top && pageY <= r.bottom;
  });

  // Hover effects
  if (key !== currentHoveredKey) {
    if (currentHoveredKey) currentHoveredKey.classList.remove("hovered");
    if (key) key.classList.add("hovered");
    currentHoveredKey = key;
  }

  if (key) startHover(key);
  else cancelHover();
}

function startHover(key) {
  if (activeKey === key) return;
  cancelHover();
  activeKey = key;
  key.classList.add("highlight");
  dwellTimer = setTimeout(() => pressKey(key), DWELL_TIME);
}

function cancelHover() {
  if (activeKey) activeKey.classList.remove("highlight");
  if (currentHoveredKey) currentHoveredKey.classList.remove("hovered");
  clearTimeout(dwellTimer);
  dwellTimer = null;
  activeKey = null;
  currentHoveredKey = null;
}

function pressKey(key) {
  const val = key.textContent.trim();
  if (key.classList.contains("delete")) {
    textContainer.textContent = textContainer.textContent.slice(0, -1);
  } else if (key.classList.contains("space")) {
    textContainer.textContent += " ";
  } else {
    textContainer.textContent += val;
  }

  key.classList.remove("highlight");
  activeKey = null;
  console.log(`🧠 Typed: ${val}`);
}

// === Boot sequence ===
(async () => {
  await initCamera();
  await loadModel();
  startCalibration();
})();
