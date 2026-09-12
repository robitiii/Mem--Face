'use strict';

// Sticker files live in memes/. Any format the browser can render works,
// including animated .gif. Nothing here generates or fetches images.
//
// A rule fires on one of three things:
//   - one of face-api's seven expression classes (neutral, happy, sad, angry,
//     fearful, disgusted, surprised)
//   - a metric derived from the 68 face landmarks, e.g. browRaise
//   - a boolean gesture from MediaPipe's hand landmarker, e.g. handsToFace
//
// `threshold` is an absolute score (0-1), used when uncalibrated.
// `sigma` is how many standard deviations above YOUR resting face the score
// must sit, used once calibrated. Rules without a `sigma` always use the
// absolute threshold — that is deliberate for `neutral`, where your baseline
// is already high and "above baseline" is not a meaningful trigger.
const RULES = [
  { expression: 'surprised', threshold: 0.7, sigma: 3.0, src: 'memes/confused.jpg' },
  { expression: 'happy', threshold: 0.8, sigma: 3.0, src: 'memes/laughing.png' },
  { expression: 'angry', threshold: 0.7, sigma: 3.0, src: 'memes/angry.png' },
  { expression: 'fearful', threshold: 0.5, sigma: 2.5, src: 'memes/fearful.jpg' },
  { expression: 'disgusted', threshold: 0.5, sigma: 2.5, src: 'memes/disgusted.jpg' },
  { expression: 'sad', threshold: 0.6, sigma: 3.0, src: 'memes/sad.jpg' },
  // "Bored" is sustained neutral, not a momentary face. Neutral is also your
  // resting score, so this rule needs a long dwell and a long cooldown or it
  // would fire on a loop any time you sit still.
  { expression: 'neutral', threshold: 0.9, src: 'memes/bored.jpg', ticks: 40, cooldownMs: 20000 },
  // Brow raise is measured from the 68 landmarks, not from the expression net.
  // It has no `threshold`, only a `sigma`, which means it cannot fire until you
  // calibrate — there is no sensible absolute value for "eyebrows up", it is
  // only meaningful relative to where your brows normally sit.
  { expression: 'browRaise', sigma: 2.5, src: 'memes/eyebrow.jpg', cooldownMs: 2000 },
  // Gesture rules are boolean rather than scored, and come from MediaPipe's
  // hand landmarker rather than face-api. A deliberate gesture outranks any
  // incidental expression — see GESTURE_STRENGTH.
  { gesture: 'handsToFace', src: 'memes/hands.png', cooldownMs: 2500 },
];

/** Ranked above every expression rule: putting your hands up is intentional. */
const GESTURE_STRENGTH = 10;

const EXPRESSIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'];

/** Landmark-derived signals that ride alongside the expression scores. */
const METRICS = ['browRaise'];

/** Everything calibration records a mean and sigma for. */
const CHANNELS = [...EXPRESSIONS, ...METRICS];

const CONFIG = {
  modelUri: 'models',
  tickMs: 160, // detection cadence
  ticksToTrigger: 2, // must stay above threshold this many ticks in a row
  cooldownMs: 1500, // minimum gap between triggers
  visibleMs: 1800, // auto-hide delay
  boxScale: 1.2, // sticker width relative to face box (no-landmark fallback)
  eyeScale: 2.6, // sticker width relative to interocular distance (with landmarks)
  smoothing: 0.35, // EMA weight for new position samples; lower = smoother, laggier
  missTolerance: 5, // dropped detections tolerated before state resets
  calibrationMs: 6000, // how long to hold a neutral face
  minCalibrationSamples: 20, // below this the baseline is too noisy to trust
  sigmaFloor: 0.01, // stops a near-zero spread producing absurd z-scores
  sigmaCeil: 0.3,
  sigmaMinScore: 0.15, // absolute floor, so tiny-but-unusual scores cannot fire
  logScores: true, // console.log expression scores (throttled)
  logEveryMs: 1000,

  // Hand tracking. The wasm runtime and model are ~20MB together, so this
  // loads in the background after the face pipeline is already running.
  handWasmPath: 'vendor/tasks-vision/wasm',
  handModelPath: 'models/hand_landmarker.task',
  handBundlePath: './vendor/tasks-vision/vision_bundle.mjs',
  handEveryTicks: 2, // run hand detection on every Nth tick to limit the cost
  handContactPoints: 4, // landmarks inside the face box before a hand counts as touching
  handsRequired: 2, // how many hands must be touching — 2 means both, 1 means either
  handFaceMargin: 0.15, // grow the face box by this fraction when testing contact
};

const STORAGE_KEY = 'meme-face-baseline';

const video = document.getElementById('video');
const anchor = document.getElementById('meme-anchor');
const meme = document.getElementById('meme');
const statusEl = document.getElementById('status');
const controlsEl = document.getElementById('controls');
const calibrateButton = document.getElementById('calibrate');
const clearCalibrationButton = document.getElementById('clear-calibration');
const calibrationStatusEl = document.getElementById('calibration-status');

const state = {
  streak: { expression: null, count: 0 },
  lastTriggerAt: 0,
  lastGeometry: null,
  smoothed: null,
  misses: 0,
  hideTimer: null,
  lastLogAt: 0,
  baseline: null,
  calibrating: null,
  handLandmarker: null,
  gestures: { handsToFace: false },
  ticks: 0,
  lastHandTimestamp: 0,
};

function setStatus(kind, message) {
  statusEl.textContent = message;
  statusEl.className = 'status status--' + kind;
}

async function loadModels() {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(CONFIG.modelUri),
    faceapi.nets.faceLandmark68Net.loadFromUri(CONFIG.modelUri),
    faceapi.nets.faceExpressionNet.loadFromUri(CONFIG.modelUri),
  ]);
}

async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  // Wait for real dimensions — videoWidth/videoHeight are 0 until metadata lands.
  if (!video.videoWidth) {
    await new Promise((resolve) => {
      video.addEventListener('loadedmetadata', resolve, { once: true });
    });
  }
  await video.play();
}

function centroid(points) {
  const total = points.reduce(
    (acc, point) => ({ x: acc.x + point.x, y: acc.y + point.y }),
    { x: 0, y: 0 }
  );
  return { x: total.x / points.length, y: total.y / points.length };
}

/**
 * Where the sticker should sit, in the video's natural pixel coordinates.
 *
 * With landmarks we get a steadier centre (midway between the eyes and the
 * mouth), a scale that does not jitter along with the detector box
 * (interocular distance), and a tilt angle. Without them, fall back to the box.
 */
function faceGeometry(result) {
  const box = result.detection.box;
  const landmarks = result.landmarks;

  if (!landmarks) {
    return {
      centerX: box.x + box.width / 2,
      centerY: box.y + box.height / 2,
      width: box.width * CONFIG.boxScale,
      angle: 0,
    };
  }

  const leftEye = centroid(landmarks.getLeftEye());
  const rightEye = centroid(landmarks.getRightEye());
  const mouth = centroid(landmarks.getMouth());
  const eyeMid = { x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2 };

  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;

  return {
    centerX: (eyeMid.x + mouth.x) / 2,
    centerY: (eyeMid.y + mouth.y) / 2,
    width: Math.hypot(dx, dy) * CONFIG.eyeScale,
    angle: (Math.atan2(dy, dx) * 180) / Math.PI,
  };
}

/**
 * How far the eyebrows sit above the eyes, as a fraction of interocular
 * distance.
 *
 * Dividing by eye spacing makes it scale-invariant: leaning toward the camera
 * moves every landmark but leaves the ratio alone. There is no useful absolute
 * value here — brow height varies enormously between faces — so this is only
 * ever read as a deviation from your calibrated baseline.
 */
function browRaise(landmarks) {
  if (!landmarks) return 0;

  const leftEye = centroid(landmarks.getLeftEye());
  const rightEye = centroid(landmarks.getRightEye());
  const interocular = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
  if (!interocular) return 0;

  const browY = (centroid(landmarks.getLeftEyeBrow()).y + centroid(landmarks.getRightEyeBrow()).y) / 2;
  const eyeY = (leftEye.y + rightEye.y) / 2;

  // y grows downward, so eyes minus brows is positive and grows as brows lift.
  return (eyeY - browY) / interocular;
}

/**
 * Exponential moving average over the raw geometry. CSS transitions already
 * hide the gaps between ticks, but they interpolate toward a jittery target;
 * this damps the jitter itself. Tilt stays near zero so it needs no wrap
 * handling.
 */
function smoothGeometry(geometry) {
  if (!state.smoothed) {
    state.smoothed = { ...geometry };
    return state.smoothed;
  }

  const smoothed = state.smoothed;
  const weight = CONFIG.smoothing;
  smoothed.centerX += (geometry.centerX - smoothed.centerX) * weight;
  smoothed.centerY += (geometry.centerY - smoothed.centerY) * weight;
  smoothed.width += (geometry.width - smoothed.width) * weight;
  smoothed.angle += (geometry.angle - smoothed.angle) * weight;
  return smoothed;
}

// ---------------------------------------------------------------------------
// Hand tracking
//
// face-api has no hand model, so gestures come from MediaPipe's hand
// landmarker. It is a separate runtime and ~20MB of wasm plus weights, so it
// loads in the background and the demo runs fine without it.
// ---------------------------------------------------------------------------

async function initHandTracking() {
  const vision = await import(CONFIG.handBundlePath);
  const fileset = await vision.FilesetResolver.forVisionTasks(CONFIG.handWasmPath);

  state.handLandmarker = await vision.HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: CONFIG.handModelPath },
    runningMode: 'VIDEO',
    numHands: 2,
  });
}

/**
 * True when `handsRequired` hands each have enough landmarks inside the face
 * box.
 *
 * Landmarks arrive normalised to 0-1, so they scale to the video's natural
 * pixels. Requiring several points per hand stops a fingertip drifting past
 * your cheek from counting, and requiring two hands stops a single hand resting
 * on your chin from firing the sticker.
 */
function handsTouchingFace(handResult, box) {
  if (!handResult || !handResult.landmarks) return false;

  const marginX = box.width * CONFIG.handFaceMargin;
  const marginY = box.height * CONFIG.handFaceMargin;
  const left = box.x - marginX;
  const right = box.x + box.width + marginX;
  const top = box.y - marginY;
  const bottom = box.y + box.height + marginY;

  let handsInContact = 0;

  for (const hand of handResult.landmarks) {
    let inside = 0;
    for (const point of hand) {
      const x = point.x * video.videoWidth;
      const y = point.y * video.videoHeight;
      if (x >= left && x <= right && y >= top && y <= bottom) inside += 1;
      if (inside >= CONFIG.handContactPoints) {
        handsInContact += 1;
        break;
      }
    }
  }

  return handsInContact >= CONFIG.handsRequired;
}

function detectGestures(box) {
  if (!state.handLandmarker) return;

  // MediaPipe rejects timestamps that do not strictly increase.
  const timestamp = Math.max(performance.now(), state.lastHandTimestamp + 1);
  state.lastHandTimestamp = timestamp;

  const handResult = state.handLandmarker.detectForVideo(video, timestamp);
  state.gestures.handsToFace = handsTouchingFace(handResult, box);
}

/** Centre of frame, for manual button presses before any face has been seen. */
function fallbackGeometry() {
  return {
    centerX: video.videoWidth / 2,
    centerY: video.videoHeight / 2,
    width: video.videoWidth * 0.4,
    angle: 0,
  };
}

/** Map natural-resolution geometry onto the video's rendered on-screen size. */
function positionMeme(geometry) {
  if (!video.videoWidth || !video.videoHeight) return;

  const scaleX = video.clientWidth / video.videoWidth;
  const scaleY = video.clientHeight / video.videoHeight;

  anchor.style.width = geometry.width * scaleX + 'px';
  anchor.style.left = video.clientWidth - geometry.centerX * scaleX + 'px';
  anchor.style.top = geometry.centerY * scaleY + 'px';
  // The preview is mirrored, so on-screen tilt runs opposite the detected tilt.
  anchor.style.setProperty('--meme-rotate', -geometry.angle + 'deg');
}

function showMeme(rule, geometry) {
  if (meme.getAttribute('src') !== rule.src) {
    meme.src = rule.src;
  }

  // Place it with gliding off, so a new sticker pops into position rather than
  // sliding in from wherever the previous one happened to be sitting.
  anchor.classList.add('is-snapping');
  positionMeme(geometry);
  void anchor.offsetWidth; // force reflow so the snap lands before transitions resume
  anchor.classList.remove('is-snapping');

  meme.classList.add('is-visible');
  clearTimeout(state.hideTimer);
  state.hideTimer = setTimeout(hideMeme, CONFIG.visibleMs);
}

function hideMeme() {
  meme.classList.remove('is-visible');
}

/** How far a live score sits above your calibrated resting face, in sigmas. */
function zScore(expression, value) {
  const mean = state.baseline.mean[expression] ?? 0;
  const sigma = state.baseline.sigma[expression] ?? CONFIG.sigmaCeil;
  return (value - mean) / sigma;
}

/** Rules are keyed by expression or gesture, whichever they use. */
function ruleKey(rule) {
  return rule.expression ?? rule.gesture;
}

/**
 * Best matching rule, or null.
 *
 * Rules are ranked by how far past their own bar they sit, not by raw score —
 * otherwise an absolute rule and a sigma rule could not be compared fairly.
 */
function matchRule(expressions) {
  let best = null;

  for (const rule of RULES) {
    if (rule.gesture) {
      if (!state.gestures[rule.gesture]) continue;
      if (!best || GESTURE_STRENGTH > best.strength) {
        best = { rule, strength: GESTURE_STRENGTH };
      }
      continue;
    }

    const score = expressions[rule.expression] ?? 0;
    let strength;

    if (state.baseline && rule.sigma != null) {
      if (score < CONFIG.sigmaMinScore) continue;
      const z = zScore(rule.expression, score);
      if (z < rule.sigma) continue;
      strength = z / rule.sigma;
    } else if (rule.threshold != null) {
      if (score <= rule.threshold) continue;
      strength = score / rule.threshold;
    } else {
      continue; // sigma-only rule with nothing calibrated to compare against
    }

    if (!best || strength > best.strength) {
      best = { rule, strength };
    }
  }

  return best;
}

function evaluate(expressions, geometry) {
  const match = matchRule(expressions);

  if (!match) {
    state.streak = { expression: null, count: 0 };
    return;
  }

  // Debounce: the same expression has to clear its bar on consecutive ticks.
  if (state.streak.expression === ruleKey(match.rule)) {
    state.streak.count += 1;
  } else {
    state.streak = { expression: ruleKey(match.rule), count: 1 };
  }

  const ticksNeeded = match.rule.ticks ?? CONFIG.ticksToTrigger;
  if (state.streak.count < ticksNeeded) return;

  const now = performance.now();
  const cooldown = match.rule.cooldownMs ?? CONFIG.cooldownMs;
  if (now - state.lastTriggerAt < cooldown) return;

  state.lastTriggerAt = now;
  state.streak.count = 0; // start the next streak fresh after firing
  showMeme(match.rule, geometry);
}

// ---------------------------------------------------------------------------
// Calibration
//
// face-api's scores sit at very different resting levels from person to
// person — some faces idle at sad 0.3, some never produce disgust above 0.2.
// Fixed thresholds therefore either never fire or fire constantly. Recording
// your own neutral face lets every rule be expressed relative to it instead.
// ---------------------------------------------------------------------------

function startCalibration() {
  if (state.calibrating) return;
  hideMeme();
  state.calibrating = { samples: [], startedAt: performance.now() };
  calibrationStatusEl.textContent = '';
  setStatus('loading', 'Calibrating — hold a neutral face…');
}

function collectCalibrationSample(expressions) {
  const session = state.calibrating;
  session.samples.push(expressions);

  const elapsed = performance.now() - session.startedAt;
  if (elapsed >= CONFIG.calibrationMs) {
    finishCalibration();
    return;
  }

  const remaining = Math.ceil((CONFIG.calibrationMs - elapsed) / 1000);
  setStatus('loading', 'Calibrating — hold a neutral face… ' + remaining + 's');
}

function finishCalibration() {
  const samples = state.calibrating.samples;
  state.calibrating = null;

  if (samples.length < CONFIG.minCalibrationSamples) {
    setStatus('error', 'Calibration failed — your face was not detected often enough.');
    calibrationStatusEl.textContent =
      'Only ' + samples.length + ' usable samples. Try better lighting, or move closer to the camera.';
    return;
  }

  const mean = {};
  const sigma = {};

  for (const name of CHANNELS) {
    const values = samples.map((sample) => sample[name] ?? 0);
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - average) ** 2, 0) / values.length;

    mean[name] = average;
    sigma[name] = Math.min(Math.max(Math.sqrt(variance), CONFIG.sigmaFloor), CONFIG.sigmaCeil);
  }

  state.baseline = { mean, sigma, samples: samples.length, made: new Date().toISOString() };
  saveBaseline();
  setStatus('ready', 'Calibrated — pull a face.');
  reportBaseline();
}

/** The two ways a calibration goes wrong: done mid-expression, or while moving. */
function calibrationWarnings(baseline) {
  const warnings = [];

  if (baseline.mean.happy > 0.25) warnings.push('you were smiling');
  if (baseline.mean.surprised > 0.25) warnings.push('your brows looked raised');
  if (baseline.mean.neutral < 0.5) warnings.push('your resting face did not read as neutral');

  const pinned = CHANNELS.filter((name) => baseline.sigma[name] >= CONFIG.sigmaCeil).length;
  if (pinned >= 3) warnings.push('you moved a lot, so the baseline is loose');

  return warnings;
}

function reportBaseline() {
  const baseline = state.baseline;

  if (!baseline) {
    calibrationStatusEl.textContent = 'Not calibrated — using absolute thresholds.';
    return;
  }

  const summary = CHANNELS.map(
    (name) => name + ' ' + baseline.mean[name].toFixed(2) + '±' + baseline.sigma[name].toFixed(2)
  ).join('   ');

  const warnings = calibrationWarnings(baseline);
  calibrationStatusEl.textContent =
    'Baseline (' + baseline.samples + ' samples): ' + summary +
    (warnings.length ? '\nHeads up: ' + warnings.join('; ') + '. Recalibrate if triggers feel off.' : '');

  console.log('Calibrated baseline', baseline);
}

function saveBaseline() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.baseline));
  } catch (error) {
    console.warn('Could not save baseline:', error);
  }
}

function loadBaseline() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) state.baseline = JSON.parse(raw);
  } catch (error) {
    console.warn('Could not load baseline:', error);
  }
}

function clearCalibration() {
  state.baseline = null;
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.warn('Could not clear baseline:', error);
  }
  reportBaseline();
}

/** Manual override row — one button per rule, for when a class will not fire. */
function buildControls() {
  for (const rule of RULES) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'control';
    button.textContent = rule.gesture ? 'hands to face' : rule.expression;
    button.addEventListener('click', () => {
      // Skip the debounce and cooldown, but stamp the trigger time so the
      // detector does not immediately replace what you just asked for.
      state.lastTriggerAt = performance.now();
      state.streak = { expression: null, count: 0 };
      showMeme(rule, state.lastGeometry ?? fallbackGeometry());
    });
    controlsEl.appendChild(button);
  }

  calibrateButton.addEventListener('click', startCalibration);
  clearCalibrationButton.addEventListener('click', clearCalibration);
}

function logScores(expressions) {
  const now = performance.now();
  if (now - state.lastLogAt < CONFIG.logEveryMs) return;
  state.lastLogAt = now;

  const scores = Object.entries(expressions)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => {
      const base = name + ': ' + value.toFixed(2);
      if (!state.baseline) return base;
      return base + ' (' + zScore(name, value).toFixed(1) + 'σ)';
    })
    .join('  ');

  console.log(scores);
}

async function tick() {
  const detectorOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 224,
    scoreThreshold: 0.5,
  });

  const result = await faceapi
    .detectSingleFace(video, detectorOptions)
    .withFaceLandmarks()
    .withFaceExpressions();

  if (!result) {
    // Tolerate brief dropouts rather than throwing away a building streak or
    // snapping the sticker on re-acquisition.
    state.misses += 1;
    if (state.misses > CONFIG.missTolerance) {
      state.streak = { expression: null, count: 0 };
      state.smoothed = null;
    }
    return;
  }

  state.misses = 0;
  state.ticks += 1;

  const geometry = smoothGeometry(faceGeometry(result));
  state.lastGeometry = geometry;

  // Landmark-derived metrics join the expression scores so they calibrate and
  // match through exactly the same machinery.
  const channels = { ...result.expressions, browRaise: browRaise(result.landmarks) };

  if (state.calibrating) {
    collectCalibrationSample(channels);
    return;
  }

  // Hand detection is the expensive half, so it runs at a fraction of the
  // face cadence and its last verdict carries over between runs.
  if (state.handLandmarker && state.ticks % CONFIG.handEveryTicks === 0) {
    try {
      detectGestures(result.detection.box);
    } catch (error) {
      console.error('Hand detection failed:', error);
    }
  }

  if (CONFIG.logScores) logScores(channels);
  evaluate(channels, geometry);

  // Keep an already-visible sticker glued to the face as it moves.
  if (meme.classList.contains('is-visible')) {
    positionMeme(geometry);
  }
}

/**
 * Self-scheduling loop rather than setInterval: a detection pass can take
 * longer than the tick, and setInterval would stack overlapping runs.
 */
async function runLoop() {
  for (;;) {
    const startedAt = performance.now();
    try {
      await tick();
    } catch (error) {
      console.error('Detection failed:', error);
    }
    const remaining = CONFIG.tickMs - (performance.now() - startedAt);
    if (remaining > 0) {
      await new Promise((resolve) => setTimeout(resolve, remaining));
    }
  }
}

function describeCameraError(error) {
  switch (error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera access was denied. Allow it in your browser’s site settings and reload.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera found. Plug one in and reload.';
    case 'NotReadableError':
      return 'The camera is already in use by another app. Close it and reload.';
    default:
      return 'Could not start the camera: ' + error.message;
  }
}

async function main() {
  if (location.protocol === 'file:') {
    setStatus('error', 'Open this page over http:// — models cannot be fetched from file://.');
    return;
  }

  buildControls();
  loadBaseline();
  reportBaseline();

  // A missing sticker is worth flagging, but it should not stop the demo.
  meme.addEventListener('error', () => {
    if (meme.getAttribute('src')) {
      setStatus('error', 'Missing sticker: ' + meme.getAttribute('src'));
    }
  });

  // Camera first: if the models fail to load, you still have a live preview
  // and working manual buttons rather than a dead page.
  try {
    setStatus('loading', 'Waiting for camera permission…');
    await startWebcam();
  } catch (error) {
    console.error(error);
    setStatus('error', describeCameraError(error));
    return;
  }

  try {
    setStatus('loading', 'Loading models…');
    await loadModels();
  } catch (error) {
    console.error(error);
    setStatus('error', 'Could not load models from /' + CONFIG.modelUri + '. Buttons still work.');
    return;
  }

  setStatus('ready', 'Ready — pull a face. (loading hand tracking…)');
  runLoop();

  // Deliberately not awaited: ~20MB of wasm and weights should not hold up the
  // demo, and a failure here must not take the expression pipeline down.
  initHandTracking().then(
    () => setStatus('ready', 'Ready — pull a face, or put your hands up.'),
    (error) => {
      console.error('Hand tracking unavailable:', error);
      setStatus('ready', 'Ready — pull a face. (hand tracking failed to load)');
    }
  );
}

main();
