const MODEL_PATH = './best.onnx';
const CLASSES = ['id_card', 'lanyard']; // adjust to your class names
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.45;
const NO_ID_GRACE_PERIOD = 2000; // ms before alarm

let session = null;
let videoStream = null;
let isProcessing = false;
let lastIdTime = Date.now();
let alarmPlaying = false;
let lastFrameTime = performance.now();

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const alertEl = document.getElementById('alert');
const alarm = document.getElementById('alarm');

async function loadModel() {
  statusEl.textContent = 'Loading model...';
  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm']
  });
  statusEl.textContent = 'System Ready';
}

loadModel();

document.getElementById('startWebcam').onclick = async () => {
  videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = videoStream;
  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detectLoop();
  };
};

document.getElementById('stopWebcam').onclick = () => {
  if (videoStream) videoStream.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  stopAlarm();
};

document.getElementById('imageInput').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = async () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const detections = await detect(canvas);
    drawDetections(detections);
    updateStatus(detections);
  };
  img.src = URL.createObjectURL(file);
};

document.getElementById('videoInput').onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.play();
  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    detectLoop();
  };
};

async function detectLoop() {
  if (!session) return;
  if (video.paused || video.ended) return;
  if (!isProcessing) {
    isProcessing = true;
    const detections = await detect(video);
    drawDetections(detections);
    updateStatus(detections);
    isProcessing = false;
    const now = performance.now();
    const fps = (1000 / (now - lastFrameTime)).toFixed(1);
    document.getElementById('fps').textContent = fps;
    lastFrameTime = now;
  }
  requestAnimationFrame(detectLoop);
}

async function detect(source) {
  const tensor = preprocess(source);
  const results = await session.run({ images: tensor });
  const output = results[Object.keys(results)[0]];
  return postprocess(output, source.videoWidth || source.width, source.videoHeight || source.height);
}

function preprocess(source) {
  const off = document.createElement('canvas');
  off.width = INPUT_SIZE;
  off.height = INPUT_SIZE;
  const c = off.getContext('2d');
  c.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const imageData = c.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = imageData;
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    float32[i] = data[i * 4] / 255;
    float32[i + INPUT_SIZE * INPUT_SIZE] = data[i * 4 + 1] / 255;
    float32[i + 2 * INPUT_SIZE * INPUT_SIZE] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor('float32', float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function postprocess(output, origW, origH) {
  // YOLOv8 output: [1, 4+nc, 8400]
  const data = output.data;
  const dims = output.dims;
  const numClasses = dims[1] - 4;
  const numBoxes = dims[2];
  const boxes = [];

  for (let i = 0; i < numBoxes; i++) {
    let maxScore = 0, classId = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > maxScore) { maxScore = score; classId = c; }
    }
    if (maxScore < CONF_THRESHOLD) continue;
    const cx = data[i];
    const cy = data[numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];
    const x1 = (cx - w / 2) / INPUT_SIZE * origW;
    const y1 = (cy - h / 2) / INPUT_SIZE * origH;
    const x2 = (cx + w / 2) / INPUT_SIZE * origW;
    const y2 = (cy + h / 2) / INPUT_SIZE * origH;
    boxes.push({ x1, y1, x2, y2, score: maxScore, classId });
  }
  return nms(boxes, IOU_THRESHOLD);
}

function nms(boxes, iouThresh) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const b = boxes.shift();
    keep.push(b);
    boxes = boxes.filter(o => iou(b, o) < iouThresh);
  }
  return keep;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function drawDetections(detections) {
  if (video.srcObject || video.src) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  detections.forEach(d => {
    const color = d.classId === 0 ? '#00FF00' : '#FFD700';
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    ctx.fillStyle = color;
    ctx.font = '18px Arial';
    const label = `${CLASSES[d.classId]} ${(d.score * 100).toFixed(0)}%`;
    ctx.fillRect(d.x1, d.y1 - 22, ctx.measureText(label).width + 10, 22);
    ctx.fillStyle = '#000';
    ctx.fillText(label, d.x1 + 5, d.y1 - 5);
  });
}

function updateStatus(detections) {
  const ids = detections.filter(d => CLASSES[d.classId] === 'id_card').length;
  const lanyards = detections.filter(d => CLASSES[d.classId] === 'lanyard').length;
  document.getElementById('idCount').textContent = ids;
  document.getElementById('lanyardCount').textContent = lanyards;

  if (ids > 0) {
    lastIdTime = Date.now();
    statusEl.className = 'status-ok';
    statusEl.textContent = '✅ ACCESS GRANTED - ID Verified';
    alertEl.style.display = 'none';
    stopAlarm();
  } else {
    const elapsed = Date.now() - lastIdTime;
    if (elapsed > NO_ID_GRACE_PERIOD) {
      statusEl.className = 'status-alert';
      statusEl.textContent = '🚫 ACCESS DENIED - No ID Card!';
      alertEl.style.display = 'block';
      playAlarm();
    }
  }
}

function playAlarm() {
  if (!alarmPlaying) {
    alarm.play().catch(() => {});
    alarmPlaying = true;
  }
}

function stopAlarm() {
  alarm.pause();
  alarm.currentTime = 0;
  alarmPlaying = false;
}
