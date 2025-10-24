// Simulate gaze hover selection for 2 seconds
const keys = document.querySelectorAll(".key");
const textContainer = document.getElementById("textContainer");
const DWELL_TIME = 2000; // 2 seconds
let dwellTimer = null;
let activeKey = null;

// Simulated "gaze" detection — you’ll later replace this
// with your red-dot gaze coordinates or model output.
keys.forEach((key) => {
  key.addEventListener("mouseenter", () => startHover(key));
  key.addEventListener("mouseleave", cancelHover);
});

function startHover(key) {
  cancelHover();
  activeKey = key;
  key.classList.add("highlight");

  dwellTimer = setTimeout(() => {
    pressKey(key);
  }, DWELL_TIME);
}

function cancelHover() {
  if (activeKey) activeKey.classList.remove("highlight");
  clearTimeout(dwellTimer);
  activeKey = null;
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
}
