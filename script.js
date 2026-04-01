console.log("--- PISCES SYSTEM STARTING ---");

// ── FaceMesh Init ──────────────────────────────
const faceMesh = new FaceMesh({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.7,
});

// ── Capture the current nudged/zoomed face frame ──
// Uses the global imgTranslateX / imgTranslateY / imgScale set by the HTML.
function captureFaceFrame() {
  const canvas = document.getElementById("capture-canvas");
  const ctx = canvas.getContext("2d");
  const frame = document.getElementById("faceFrameContainer");
  const img = document.getElementById("previewImg");

  canvas.width = frame.offsetWidth;
  canvas.height = frame.offsetHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  // Reconstruct the same transform the CSS applies so MediaPipe
  // sees exactly what the user aligned inside the oval.
  ctx.translate(
    canvas.width / 2 + imgTranslateX,
    canvas.height / 2 + imgTranslateY
  );
  ctx.scale(imgScale, imgScale);
  ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
  ctx.restore();

  return canvas;
}

// ── Grab base64 from whichever canvas was last used ──────────────────
function getCanvasBase64() {
  const canvas = document.getElementById("capture-canvas");
  try {
    return canvas.toDataURL("image/jpeg", 0.85).split(",")[1];
  } catch (e) {
    console.warn("Could not capture base64 image:", e);
    return "";
  }
}

// ── Override analyzeUploadedImage (defined in the HTML) ───────────────
// Run MediaPipe first; the onResults callback fires runAnalysis() after.
window.analyzeUploadedImage = function () {
  if (!uploadedImageData) return;
  faceMesh.send({ image: captureFaceFrame() });
};

// ── Override capturePhoto (camera path) ──────────────────────────────
// Re-implement capture so it routes through MediaPipe before runAnalysis().
window.capturePhoto = function () {
  if (!stream) return;

  const canvas = document.getElementById("capture-canvas");
  const ctx = canvas.getContext("2d");
  const video = document.getElementById("camera-video");

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.save();
  ctx.scale(-1, 1); // un-mirror the front camera
  ctx.drawImage(video, -canvas.width, 0);
  ctx.restore();

  stopCamera();
  faceMesh.send({ image: canvas });
};

// ── FaceMesh result handler ───────────────────────────────────────────
faceMesh.onResults((results) => {
  if (results.multiFaceLandmarks?.[0]) {
    console.log("✅ Face landmarks detected.");
    window.currentLandmarks = results.multiFaceLandmarks[0];
    window.currentImageBase64 = getCanvasBase64(); // ← capture image for ViT
    console.log("Base64 length:", window.currentImageBase64?.length);
    runAnalysis(); // defined in the HTML; reads window.currentLandmarks
  } else {
    console.warn("No face detected.");
    alert(
      "No face detected in the frame.\n" +
      "Try zooming in or repositioning the image so your face fills the oval."
    );
  }
});
