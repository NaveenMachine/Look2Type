console.log("🚀 Look2Type with YOLO Pupil Detection");

// === Grab HTML elements ===
const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const typedText = document.getElementById("typed-text");

let session = null;
let modelLoaded = false;
let cameraReady = false;
let lastKey = null;
let dwellStart = 0;
let frameCount = 0;

const MODEL_SIZE = 384;

// === Step 1: Initialize Webcam ===
async function initCamera() {
  try {
    console.log("🎥 Requesting camera access...");
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onplaying = () => {
        console.log("✅ Webcam feed is live");
        cameraReady = true;
        resolve();
      };
    });
  } catch (err) {
    console.error("❌ Camera access failed:", err);
    alert("Please allow camera access in your browser to use Look2Type.");
  }
}

// === Step 2: Load ONNX Model ===
async function loadModel() {
  try {
    console.log("📦 Loading ONNX model...");
    session = await ort.InferenceSession.create("./models/best.onnx");
    modelLoaded = true;
    console.log("✅ Model loaded successfully!");
  } catch (err) {
    console.error("❌ Failed to load ONNX model:", err);
    alert("Could not load the model. Check that './models/best.onnx' exists.");
  }
}

// === Step 3: Draw and handle detections ===
function drawDetections(detections, width, height) {
  if (!detections.length) return;

  detections.forEach(([x, y, w, h, conf, cls]) => {
    if (conf < 0.3) return; // lower threshold to test visualization

    const xCenter = x * width;
    const yCenter = y * height;

    // Draw pupil dot
    ctx.beginPath();
    ctx.arc(xCenter, yCenter, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "lime";
    ctx.fill();

    // === Gaze typing logic ===
    const keyWidth = width / 6; // 6 keys (A–F)
    const keyIndex = Math.floor((xCenter / width) * 6);
    const currentKey = String.fromCharCode(65 + keyIndex);

    const now = Date.now();
    if (currentKey === lastKey) {
      if (now - dwellStart > 1000) {
        typedText.textContent += currentKey;
        dwellStart = now;
      }
    } else {
      lastKey = currentKey;
      dwellStart = now;
    }
  });
}

// === Step 4: Detection Loop ===
async function detect() {
  if (!modelLoaded || !cameraReady) {
    requestAnimationFrame(detect);
    return;
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = MODEL_SIZE;
  offscreen.height = MODEL_SIZE;
  const offctx = offscreen.getContext("2d");
  offctx.drawImage(video, 0, 0, MODEL_SIZE, MODEL_SIZE);

  const imageData = offctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const float32Data = new Float32Array(MODEL_SIZE * MODEL_SIZE * 3);
  for (let i = 0; i < MODEL_SIZE * MODEL_SIZE; i++) {
    float32Data[i] = imageData.data[i * 4] / 255.0;
    float32Data[i + MODEL_SIZE * MODEL_SIZE] = imageData.data[i * 4 + 1] / 255.0;
    float32Data[i + 2 * MODEL_SIZE * MODEL_SIZE] = imageData.data[i * 4 + 2] / 255.0;
  }

  const tensor = new ort.Tensor("float32", float32Data, [1, 3, MODEL_SIZE, MODEL_SIZE]);

  try {
    const result = await session.run({ [session.inputNames[0]]: tensor });
    const output = result[session.outputNames[0]];
    const data = output.data || output;

    // Detect shape flexibly
    let stride;
    if (output.dims.length === 3) stride = output.dims[2]; // [1, N, 6]
    else if (output.dims.length === 2) stride = output.dims[1]; // [N, 6]
    else stride = 6; // fallback

    const detections = [];
    for (let i = 0; i < data.length; i += stride) {
      detections.push(data.slice(i, i + stride));
    }

    // Log once for verification
    if (frameCount < 3) {
      console.log(`📸 Frame ${frameCount} detections:`, detections.slice(0, 2));
      frameCount++;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    drawDetections(detections, canvas.width, canvas.height);
  } catch (err) {
    console.error("❌ Inference error:", err);
  }

  requestAnimationFrame(detect);
}

// === Step 5: Run Everything ===
(async () => {
  await initCamera();
  await loadModel();
  detect();
})();
