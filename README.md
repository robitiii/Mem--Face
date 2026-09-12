# Meme Face

Your webcam catches you pulling a face, a meme lands on it.

A single HTML page that watches your webcam, classifies your expression in real
time, and drops a reaction meme over your face — tracked with 68-point facial
landmarks so it follows and tilts with your head. Screen-share the tab into a
call and your reactions become memes.

No framework, no build step, no npm, no backend. Everything runs client-side,
and the models are bundled so it works offline.

## Run it

```bash
node serve.js          # http://localhost:8080
```

Then open <http://localhost:8080> and allow camera access.

It must be served over HTTP — `file://` cannot fetch the model weights, and the
page will tell you so rather than failing silently. `localhost` counts as a
secure context, so the webcam works without any certificate setup.

Any static server works; VS Code's Live Server extension is fine too.

## How it works

Three models run per tick (~160ms), all from `models/`:

| Model | Job |
|---|---|
| `tinyFaceDetector` | finds the face bounding box |
| `faceLandmark68Net` | 68 facial landmarks for position, scale and tilt |
| `faceExpressionNet` | scores 7 expression classes |

Hand gestures come from a second runtime — MediaPipe's `HandLandmarker` —
because face-api has no hand model at all. It loads in the background after the
face pipeline is already running and runs on every other tick, so the demo
works fine (minus gestures) if it fails or is still downloading.

The sticker is positioned from the landmarks rather than the raw detection box:
its centre is the midpoint between the eye line and the mouth, its width is a
multiple of **interocular distance** (far steadier frame-to-frame than box
width), and its tilt comes from the angle of the eye line. Position, size and
tilt are run through an exponential moving average, then CSS transitions glide
between ticks.

The preview is CSS-mirrored so it reads like a normal camera; the overlay is
not (that would flip the sticker), so the detection x-coordinate and tilt angle
are flipped in code instead.

## Triggers

Rules live at the top of `script.js`:

```js
{ expression: 'surprised', threshold: 0.7, sigma: 3.0, src: 'memes/confused.jpg' }
```

Some rules read a **metric derived from the 68 landmarks** rather than an
expression score. `browRaise` is the vertical gap between brows and eyes,
divided by interocular distance so leaning toward the camera does not trigger
it. It has a `sigma` but no `threshold`, which means **it cannot fire until you
calibrate** — brow height varies too much between faces for an absolute number
to mean anything.

Gesture rules use a boolean instead of a score:

```js
{ gesture: 'handsToFace', src: 'memes/hands.png', cooldownMs: 2500 }
```

`handsToFace` fires when **both** hands each have at least 4 landmarks inside
the face box (grown by 15%). Requiring two hands stops a single hand resting on
your chin from triggering it; set `handsRequired: 1` in `CONFIG` for the looser
behaviour. Gestures outrank every expression rule — putting your hands up is
deliberate in a way that a passing expression is not.

A rule fires when its expression clears the bar on two consecutive ticks, then
a cooldown blocks re-triggering for 1.5s. Rules can override both (`ticks`,
`cooldownMs`) — the "bored" rule uses a 6.4s dwell and a 20s cooldown, because
neutral is your resting score and would otherwise fire on a loop.

### Calibration

face-api's expression scores sit at very different resting levels from person to
person — some faces idle at `sad: 0.3`, some never produce `disgusted` above
0.2. Fixed thresholds therefore either never fire or fire constantly.

Hit **Calibrate**, hold a neutral face for 6 seconds, and the page records the
mean and standard deviation of all seven scores. Rules with a `sigma` field then
trigger on standard deviations above *your* baseline instead of an absolute
number. The baseline persists in `localStorage`.

Two deliberate details:

- **`neutral` has no `sigma`.** Your baseline *is* neutral, so "above baseline
  neutral" is meaningless — that rule keeps its absolute threshold.
- **`sigmaMinScore` floor.** If your resting `disgusted` is 0.01 with near-zero
  spread, 0.05 is "4σ" but means nothing. Without the floor, calibration would
  make the weak classes fire on noise.

## Tuning

Everything lives in `CONFIG` in `script.js`:

| Key | Does |
|---|---|
| `eyeScale` | sticker size, as a multiple of eye spacing |
| `smoothing` | EMA weight; lower is smoother but laggier |
| `tickMs` | detection cadence |
| `visibleMs` | how long a sticker stays up |
| `logScores` | per-second console log of all 7 scores |

Leave `logScores` on while tuning — it prints live scores (with sigma
annotations once calibrated), which beats guessing at thresholds.

## Known limits

- **Single face only.** No multi-face detection.
- **`fearful` and `disgusted` are weak classes** and overlap heavily with
  `surprised` and `angry`. Calibration helps; it does not fully solve it.
- **~6fps.** Three models per tick. If it feels sluggish, raise `tickMs` or swap
  in the tiny landmark model via `withFaceLandmarks(true)`.
- Stickers are opaque images. Transparent PNGs look considerably better.

## Tests

```bash
node test.js
```

Runs `script.js` in a stubbed DOM and exercises the calibration statistics,
z-scores and rule matching — the parts that are hard to eyeball in a browser.

## Credits

Face detection and expression classification by
[face-api.js](https://github.com/justadudewhohacks/face-api.js). Hand tracking by
[MediaPipe Tasks Vision](https://developers.google.com/mediapipe).

The calibration approach is borrowed from
[gazijarin/itsgiving](https://github.com/gazijarin/itsgiving), which solves the
same per-person baseline problem against MediaPipe blendshapes.

Meme images in `memes/` are placeholders pulled from
[meming.world](https://en.meming.world) and Wikimedia Commons, and remain the
property of their respective owners.
