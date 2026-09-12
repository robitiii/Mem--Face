// Throwaway harness: runs the real script.js in a stubbed DOM so the
// calibration statistics and rule matching can be exercised without a browser.
const fs = require('fs');
const vm = require('vm');

function fakeEl() {
  const el = {
    style: { setProperty() {} },
    classList: { add() {}, remove() {}, contains: () => false },
    addEventListener() {},
    appendChild() {},
    getAttribute: () => null,
    setAttribute() {},
    textContent: '',
    className: '',
    clientWidth: 640,
    clientHeight: 480,
    videoWidth: 640,
    videoHeight: 480,
    offsetWidth: 0,
  };
  return el;
}

const store = {};
const ctx = {
  document: { getElementById: () => fakeEl(), createElement: () => fakeEl() },
  localStorage: {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = v; },
    removeItem: (k) => { delete store[k]; },
  },
  location: { protocol: 'http:' },
  performance: { now: () => Date.now() },
  console: { log() {}, warn() {}, error() {} },
  setTimeout, clearTimeout,
};
ctx.globalThis = ctx;

const src = fs.readFileSync('script.js', 'utf8') +
  '\n;globalThis.__t = { state, CONFIG, RULES, EXPRESSIONS, matchRule, zScore, finishCalibration,' +
  ' calibrationWarnings, handsTouchingFace, ruleKey, GESTURE_STRENGTH };';

vm.createContext(ctx);
new vm.Script(src).runInContext(ctx);

const t = ctx.__t;
let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) { pass++; console.log('  PASS  ' + name); }
  else { fail++; console.log('  FAIL  ' + name + (detail ? '  -> ' + detail : '')); }
}

// --- 1. Uncalibrated: absolute thresholds ------------------------------------
console.log('\nUncalibrated (absolute thresholds)');
t.state.baseline = null;

let m = t.matchRule({ neutral: 0.1, happy: 0.85, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0.05 });
check('happy 0.85 fires (> 0.8)', m && m.rule.expression === 'happy', m && m.rule.expression);

m = t.matchRule({ neutral: 0.3, happy: 0.75, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0 });
check('happy 0.75 does not fire (< 0.8)', m === null, m && m.rule.expression);

m = t.matchRule({ neutral: 0.95, happy: 0.01, sad: 0.01, angry: 0, fearful: 0, disgusted: 0, surprised: 0 });
check('neutral 0.95 fires (> 0.9)', m && m.rule.expression === 'neutral', m && m.rule.expression);

// --- 2. Calibration statistics ----------------------------------------------
console.log('\nCalibration statistics');
// Known series for `sad`: mean 0.30, population sd 0.10.
const sadValues = [0.2, 0.4, 0.2, 0.4, 0.3, 0.3, 0.2, 0.4, 0.2, 0.4,
                   0.3, 0.3, 0.2, 0.4, 0.2, 0.4, 0.3, 0.3, 0.2, 0.4,
                   0.3, 0.3, 0.2, 0.4];
t.state.calibrating = {
  samples: sadValues.map((v) => ({
    neutral: 0.6, happy: 0.02, sad: v, angry: 0.01,
    fearful: 0.01, disgusted: 0.01, surprised: 0.02,
  })),
  startedAt: 0,
};
t.finishCalibration();

const base = t.state.baseline;
const expectedMean = sadValues.reduce((a, b) => a + b, 0) / sadValues.length;
const expectedSd = Math.sqrt(sadValues.reduce((a, b) => a + (b - expectedMean) ** 2, 0) / sadValues.length);

check('baseline was created', !!base);
check('sad mean ~= ' + expectedMean.toFixed(4),
  Math.abs(base.mean.sad - expectedMean) < 1e-9, base && base.mean.sad);
check('sad sigma ~= ' + expectedSd.toFixed(4),
  Math.abs(base.sigma.sad - expectedSd) < 1e-9, base && base.sigma.sad);
check('sample count recorded', base.samples === sadValues.length, base && base.samples);
check('near-constant channel clamped to sigmaFloor',
  base.sigma.happy === t.CONFIG.sigmaFloor, base && base.sigma.happy);
check('baseline persisted to localStorage', !!store['meme-face-baseline']);

// --- 3. z-scores -------------------------------------------------------------
console.log('\nz-scores');
const z = t.zScore('sad', expectedMean + 3 * expectedSd);
check('value 3 sd above mean scores 3.0', Math.abs(z - 3) < 1e-9, z);
check('value at mean scores 0', Math.abs(t.zScore('sad', expectedMean)) < 1e-9, t.zScore('sad', expectedMean));

// --- 4. Calibrated matching --------------------------------------------------
console.log('\nCalibrated (sigma thresholds)');
// sad rule needs 3.0 sigma. mean 0.30, sd 0.10 -> fires at > 0.60.
m = t.matchRule({ neutral: 0.2, happy: 0.02, sad: 0.65, angry: 0.01, fearful: 0.01, disgusted: 0.01, surprised: 0.02 });
check('sad 0.65 fires (3.5 sigma)', m && m.rule.expression === 'sad', m && m.rule.expression);

m = t.matchRule({ neutral: 0.4, happy: 0.02, sad: 0.50, angry: 0.01, fearful: 0.01, disgusted: 0.01, surprised: 0.02 });
check('sad 0.50 does not fire (2.45 sigma)', m === null, m && m.rule.expression);

// disgusted baseline is ~0.01 with sigma clamped to floor 0.01, so 0.05 is
// many sigmas out but absolutely meaningless. The sigmaMinScore floor blocks it.
m = t.matchRule({ neutral: 0.8, happy: 0.02, sad: 0.1, angry: 0.01, fearful: 0.01, disgusted: 0.05, surprised: 0.01 });
check('tiny-but-unusual disgusted 0.05 blocked by sigmaMinScore', m === null, m && m.rule.expression);

m = t.matchRule({ neutral: 0.3, happy: 0.02, sad: 0.1, angry: 0.01, fearful: 0.01, disgusted: 0.55, surprised: 0.01 });
check('disgusted 0.55 fires once above sigmaMinScore', m && m.rule.expression === 'disgusted', m && m.rule.expression);

// neutral has no sigma field, so it must keep using the absolute threshold even
// when calibrated -- baseline neutral is 0.6, so a sigma rule would misfire.
m = t.matchRule({ neutral: 0.95, happy: 0.01, sad: 0.01, angry: 0.01, fearful: 0.01, disgusted: 0.005, surprised: 0.005 });
check('neutral still uses absolute threshold when calibrated',
  m && m.rule.expression === 'neutral', m && m.rule.expression);

m = t.matchRule({ neutral: 0.75, happy: 0.05, sad: 0.1, angry: 0.02, fearful: 0.02, disgusted: 0.03, surprised: 0.03 });
check('neutral 0.75 does not fire (< 0.9) despite being 1.5 sd above baseline',
  m === null, m && m.rule.expression);

// --- 5. Calibration warnings -------------------------------------------------
console.log('\nCalibration warnings');
check('smiling baseline warns',
  t.calibrationWarnings({ mean: { happy: 0.4, surprised: 0, neutral: 0.6 }, sigma: {} })
    .some((w) => w.includes('smiling')));
check('clean baseline warns about nothing',
  t.calibrationWarnings({ mean: { happy: 0.02, surprised: 0.02, neutral: 0.9 }, sigma: {} }).length === 0);

// --- 6. Hand-to-face contact -------------------------------------------------
console.log('\nHand-to-face contact');
// Video is 640x480 in the stub. Box 100,100 200x200 grows by 15% margin to
// x 70..330, y 70..330 -> normalised x 0.109..0.516, y 0.146..0.687.
const box = { x: 100, y: 100, width: 200, height: 200 };
const pt = (x, y) => ({ x, y });
const hand = (n, x, y) => Array.from({ length: n }, () => pt(x, y));

check('5 landmarks inside the box counts as contact',
  t.handsTouchingFace({ landmarks: [hand(5, 0.3, 0.4)] }, box) === true);

check('3 landmarks inside is below handContactPoints',
  t.handsTouchingFace({ landmarks: [[...hand(3, 0.3, 0.4), ...hand(5, 0.95, 0.95)]] }, box) === false);

check('hand entirely outside the box is not contact',
  t.handsTouchingFace({ landmarks: [hand(21, 0.95, 0.95)] }, box) === false);

check('second hand can satisfy contact alone',
  t.handsTouchingFace({ landmarks: [hand(21, 0.95, 0.95), hand(6, 0.3, 0.4)] }, box) === true);

check('no hands detected is not contact',
  t.handsTouchingFace({ landmarks: [] }, box) === false);

check('null hand result is not contact', t.handsTouchingFace(null, box) === false);

// --- 7. Gesture rules in matchRule ------------------------------------------
console.log('\nGesture rules');
t.state.baseline = null;
const calm = { neutral: 0.9, happy: 0.02, sad: 0.02, angry: 0.02, fearful: 0.01, disgusted: 0.01, surprised: 0.02 };

t.state.gestures.handsToFace = false;
m = t.matchRule(calm);
check('gesture rule does not fire when the gesture is absent',
  !m || m.rule.gesture !== 'handsToFace', m && t.ruleKey(m.rule));

t.state.gestures.handsToFace = true;
m = t.matchRule(calm);
check('gesture rule fires when the gesture is present',
  m && m.rule.gesture === 'handsToFace', m && t.ruleKey(m.rule));

// A big grin plus hands up: the deliberate gesture should win.
m = t.matchRule({ neutral: 0.05, happy: 0.92, sad: 0, angry: 0, fearful: 0, disgusted: 0, surprised: 0.03 });
check('gesture outranks a strongly-firing expression',
  m && m.rule.gesture === 'handsToFace', m && t.ruleKey(m.rule));

check('ruleKey reads gesture rules', t.ruleKey({ gesture: 'handsToFace' }) === 'handsToFace');
check('ruleKey reads expression rules', t.ruleKey({ expression: 'happy' }) === 'happy');

t.state.gestures.handsToFace = false;

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
