// Initialize MediaPipe Face Mesh
const faceMesh = new FaceMesh({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
}});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true, // Crucial for the 3D volume
  minDetectionConfidence: 0.5
});

// Link this to your existing upload input ID
const imageInput = document.getElementById('imageUpload'); 

imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    await faceMesh.send({image: img});
  };
});

// When MediaPipe gets the points, send to Python
faceMesh.onResults(async (results) => {
  if (results.multiFaceLandmarks && results.multiFaceLandmarks[0]) {
    const landmarks = results.multiFaceLandmarks[0];
    
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        landmarks: landmarks,
        height: 180 // Hardcoded for testing, link to UI later
      })
    });

    const r = await response.json();
    
    // Call the exact function from your HTML template
    showResults(r); 
  }
});
