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

// === Step 3: Draw detections directly over eyes ===
function drawDetections(output, width, height) {
  let detections = [];

  // Parse ONNX output flexibly
  if (Array.isArray(output)) {
    output.forEach((d) => {
      if (d instanceof Float32Array && d.length >= 6)
        detections.push(Array.from(d));
    });
  } else if (output.data) {
    const data = output.data;
    const stride = output.dims?.[2] || 6;
    for (let i = 0; i < data.length; i += stride) {
      detections.push(data.slice(i, i + stride));
    }
  }

  if (!detections.length) return;

  // Redraw frame
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(video, 0, 0, width, height);

  let sumX = 0,
    sumY = 0,
    validCount = 0;

  detections.forEach(([x, y, w, h, conf, cls]) => {
    if (conf < 0.3) return;

    // Convert from model coordinates (0–384) to video coordinates
    const xCenter = (x / MODEL_SIZE) * width;
    const yCenter = (y / MODEL_SIZE) * height;
    const boxW = (w / MODEL_SIZE) * width;
    const boxH = (h / MODEL_SIZE) * height;

    // Skip invalid coords
    if (xCenter < 0 || xCenter > width || yCenter < 0 || yCenter > height)
      return;

    // === Draw pupil dot ===
    ctx.beginPath();
    ctx.arc(xCenter, yCenter, 6, 0, 2 * Math.PI);
    ctx.fillStyle = "lime";
    ctx.shadowColor = "black";
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;

    // === Optional bounding box ===
    ctx.strokeStyle = "rgba(0,255,0,0.3)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(xCenter - boxW / 2, yCenter - boxH / 2, boxW, boxH);

    // === Confidence text ===
    ctx.fillStyle = "white";
    ctx.font = "13px monospace";
    ctx.fillText(`conf ${conf.toFixed(2)}`, xCenter + 8, yCenter - 5);

    sumX += xCenter;
    sumY += yCenter;
    validCount++;
  });

  // === Draw red averaged gaze dot ===
  if (validCount > 0) {
    const gazeX = sumX / validCount;
    const gazeY = sumY / validCount;
    ctx.beginPath();
    ctx.arc(gazeX, gazeY, 8, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();
    ctx.fillStyle = "white";
    ctx.font = "16px monospace";
    ctx.fillText(`Gaze: (${Math.round(gazeX)}, ${Math.round(gazeY)})`, 10, 25);
  }
}

// === Step 4: Continuous Detection Loop ===
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
    float32Data[i + MODEL_SIZE * MODEL_SIZE] =
      imageData.data[i * 4 + 1] / 255.0;
    float32Data[i + 2 * MODEL_SIZE * MODEL_SIZE] =
      imageData.data[i * 4 + 2] / 255.0;
  }

  const tensor = new ort.Tensor("float32", float32Data, [
    1,
    3,
    MODEL_SIZE,
    MODEL_SIZE,
  ]);

  try {
    const result = await session.run({ [session.inputNames[0]]: tensor });
    const output = result[session.outputNames[0]];
    drawDetections(output, canvas.width, canvas.height);

    if (frameCount < 5) {
      console.log(`📸 Frame ${frameCount} detections:`, output);
      frameCount++;
    }
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
