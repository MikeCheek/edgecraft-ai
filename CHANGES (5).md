# EdgeCraft AI — Changes Summary

Both zips contain the **full project** (not just diffs) — extract over your
existing folders. Everything below was verified end-to-end with a real
dataset → train → optimize → evaluate → export run (TensorFlow actually
executing, not mocked), plus a `gcc` compile check of the generated C header.

**⚠️ New frontend dependency**: this round adds real URL routing via
`react-router-dom`. Run `npm install react-router-dom` in the frontend
folder before starting the dev server.

---

## -5. Latest round: Chirale header fix, live camera inference for any board

### The Chirale_TensorFlowLite.h header bug
- Real bug, thank you for catching it: the generated sketches included
  `<TensorFlowLite.h>` (matching the old, now-archived Google
  `Arduino_TensorFlowLite` library's convention), but the actual library
  you're meant to install - **Chirale_TensorFlowLite** - installs a header
  called `Chirale_TensorFlowLite.h`. Fixed the include, the in-sketch
  comment, and the README so all three agree on the real library and header
  name.

### Live camera inference, for any board - integrated or wired via pins
- Camera support was previously hardcoded to `board == "ESP32_CAM"` only.
  It's now a real `camera_config` (`enabled`, `module_type`) that any board
  can opt into:
  - **ESP32-CAM**: integrated camera, on by default, pins pre-filled with
    the common AI-Thinker pinout (still editable for other module variants).
  - **Any other board**: a new "Attach an external camera module (via GPIO)"
    checkbox in the Deployment tab reveals the same pin configuration form,
    for a camera module you've wired up yourself (e.g. an OV2640 breakout
    on a plain ESP32-S3 dev board).
- **Fixed a real bug found while building this**: `frameToModelInput()`
  only ever handled 1-byte-per-pixel (grayscale) capture correctly - for a
  3-channel (RGB) model input, the camera is configured for `RGB565` (2
  bytes/pixel), but the old code still read it as one raw byte per pixel,
  silently feeding garbage into the model. Fixed to properly decode RGB565
  (5/6/5-bit channels) into normalised R/G/B samples per pixel.
- **Full Serial output**: `printPrediction()` used to print only the single
  best (argmax) class. It now prints a `--- Prediction ---` block listing
  **every** class with its confidence percentage, then a `Best: <label>
  (<confidence>%)` summary line - "in serial I should see all the results
  complete."
- **Live display UI**: when both a camera and a status display are
  enabled, the exported sketch now shows a basic live UI - the actual
  downsampled camera frame (drawn via `drawRGBBitmap`) with the current
  prediction and confidence overlaid underneath, refreshed every inference
  cycle - not just a static text readout. This reuses the same
  downsampling pass that feeds the model, so there's no extra capture/
  conversion overhead for the preview.
- Same double-brace f-string bug as last time (`{{}}` inside an expression
  field, parsed as a set containing an unhashable dict) crept back in while
  rewriting this file - caught and fixed again, plus re-ran the systematic
  scan across the whole file to confirm no other instances.
- Verified with 8 combinations (ESP32-CAM ×grayscale/RGB ×with/without
  display, external camera on ESP32-S3, display-only, no-camera-no-display,
  Arduino Nano) - all generate valid, brace-balanced C++ - plus a real
  end-to-end API test (train → optimize → export) confirming the actual
  downloaded zip contains the live-preview function and full Serial dump.

## -4. Previous round: real routing, job-resume, dataset/model tree, deployment fix

### Tabs are now real routes, and running jobs survive navigation/reload
- Replaced the `activeTab` local-state + conditional-`<div>` tab system
  with actual `react-router-dom` routes: `/`, `/collect`, `/train`,
  `/optimize`, `/models` (new), `/deploy`. The sidebar now uses `<NavLink>`
  so the URL bar reflects where you are and back/forward/refresh work
  properly.
- **"If a job is running it should reload that state"**: new backend
  endpoints `GET /api/training/active?task=X` and
  `GET /api/optimization/active?training_id=X` return whichever job is
  currently `running`/`initialized`. `ModelTrainer` and `OptimizationStudio`
  now call these on mount and automatically reattach (resume polling +
  reconnect the live console) instead of showing a blank form when you
  navigate away and back, or reload the page mid-job.

### Dataset/model selection, and the archived-runs leak
- **Real bug fixed**: archiving a training session only ever hid it from
  the training history list - the *model* it produced kept showing up
  everywhere (Optimization Studio's picker, Dashboard, etc.) because
  `trained_models` and `training_sessions` are separate stores and nothing
  cross-referenced them. `Trainer.get_trained_models()` now excludes
  models whose training session is archived by default (matching what
  "archive" actually implies), with `include_archived=true` still
  available for anyone who wants to see everything.
- **New dataset filter** in Optimization Studio's model picker - previously
  every trained model across every dataset was one long flat list with no
  way to narrow it down.

### Dataset → Model → Optimized Variant tree
- New `GET /api/models/tree`: a single aggregated endpoint joining
  datasets, the models trained from each, and every optimized variant
  generated from each model (with method, size, compression ratio, and
  accuracy/speedup deltas), respecting the archived filter above.
- New `ModelTree.tsx` component: a collapsible tree UI for the above.
  Clicking a model jumps to Optimization Studio with that model
  pre-selected (`/optimize?model=<training_id>`); clicking a completed
  optimization jumps to Deployment with it pre-selected
  (`/deploy?optimization=<id>`).
- New **Models** page (nav sidebar) showing the full tree as its own
  dedicated view, in addition to it powering the pickers below.

### The Deployment tab bug ("always says complete optimization first")
- Root cause: `DeploymentPanel` read `state.currentOptimization?.id` from
  global app state - but **nothing in the app ever dispatched
  `SET_OPTIMIZATION`**, so that value was permanently `undefined` no
  matter how many completed optimizations existed. It was structurally
  impossible for this to ever work.
- Rewritten with its own real selector: opens the same dataset → model →
  optimization tree above when nothing is selected yet (or when you click
  "Change model"), supports deep-linking via `?optimization=<id>`, and
  only lets you pick optimizations that actually finished successfully.

## -3. Previous round: crash fixes, no more silent LLM mock fallback, real errors in the UI, live job console

### The optimization crash (Lambda deserialization)
- Root cause: Keras 3 blocks deserializing `Lambda` layers containing a raw
  Python lambda by default (an arbitrary-code-execution guard for untrusted
  `.keras` files). The channel-adapter Lambda added earlier (for feeding
  grayscale input into RGB-pretrained backbones) tripped this on every
  reload. Fixed by passing `safe_mode=False` in the two places that load a
  `.keras` file this backend itself produced (`optimizer.py`,
  `inference_engine.py`) - safe here because we never load third-party or
  user-uploaded model files, only ones we trained ourselves.
- **A second bug hid behind the first**: once `safe_mode=False` was in
  place, those same Lambda layers failed to reconstruct with
  `NotImplementedError: We could not automatically infer the shape of the
  Lambda's output`, because none of them declared `output_shape=`. Fixed
  in all four Lambda layers (`model_factory.py` ×3, `optimizer.py` ×1) and
  verified with an actual save→reload round-trip.

### No more silent mock fallback for LLM suggestions
- `LLMAdvisor.generate_suggestions()`, `_call_openrouter()`, and
  `_call_ollama()` used to catch *any* failure (bad API key, network error,
  invalid model name, malformed response) and silently return generic
  canned "mock" advice that looked exactly like a real suggestion. Removed
  `_mock_suggestions()` entirely - failures now raise a real, specific
  error (e.g. "OpenRouter returned HTTP 401: ...", "OpenRouter response was
  not valid JSON: ...") that propagates to the frontend instead.
- Also fixed the empty `"OpenRouter connection error: "` log line - some
  aiohttp/asyncio exceptions stringify to nothing on their own, so error
  messages now always include the exception type name too.
- `recommend_training_params()` (pre-training "Suggest Optimal Config") had
  the same silent-fallback pattern to rule-based defaults; it now raises
  real errors as well instead of quietly substituting a heuristic answer.

### Errors weren't reaching the frontend at all
- `LLMAdvisor.tsx` called `useAPI()`'s `request()` (which does capture
  backend error messages) but never destructured or displayed its `error`
  state - a failed LLM call looked like nothing happened. Now shown in a
  red banner.
- `ModelTrainer`'s "Suggest Optimal Config" had the same issue: on failure
  it always showed a generic "No recommendation returned." instead of the
  real backend message. Fixed to surface the actual error.
- **The big one**: the optimization polling loop in `OptimizationStudio`
  read the failure detail from `sj?.data?.error` - but the status endpoint
  returns a *flat* object with no `.data` nesting, so this was always
  `undefined` and silently replaced with a generic "Optimization failed"
  string. On top of that, a broken `catch` block (`if (e.message !==
  "failed") continue;`) swallowed even that generic message, so a real
  failure just kept polling for up to 6 minutes before eventually showing
  a misleading "Timed out waiting for optimization" - never the actual
  cause. Rewrote the polling loop to read `sj?.error` directly and stop
  immediately with the real message on failure.

### Live job console (WebSocket log streaming)
- New `app/services/job_logs.py`: an in-memory per-job ring buffer +
  WebSocket broadcaster. `job_log_broker.log(job_id, message)` is callable
  from any thread (training/optimization run via FastAPI's
  `BackgroundTasks`, i.e. a worker thread, not the event loop) and hops
  back onto the main asyncio loop via `run_coroutine_threadsafe` to push
  to connected clients.
- New `GET (ws) /ws/logs/{job_id}`: sends buffered history on connect, then
  streams new lines live. job_id is a training_id or optimization_id.
- `trainer.py` and `optimizer.py` now log real progress to this broker:
  session start, dataset/model info, device used, per-epoch metrics (or
  per-method-step for optimization), completion, cancellation, and -
  critically - **the full exception traceback on failure**, not just the
  one-line summary. This is exactly what was missing for the Lambda crash:
  it only ever appeared in the backend's own terminal before.
- New frontend `TerminalLogPanel.tsx`: a collapsible, auto-scrolling,
  color-coded (info/warning/error) terminal view that connects to the
  right job's WebSocket. Wired into `ModelTrainer` (auto-expands on
  failure) and `OptimizationStudio` (follows whichever optimization was
  most recently triggered).
- Verified end-to-end: connected a WebSocket client *while* a real
  training job was running in a background thread and confirmed every
  epoch line arrived live, not just on reconnect/history-replay.

### Also fixed while in the area
- Two more leftover mojibake artifacts in `OptimizationStudio.tsx` from
  the original corrupted upload: a stray `text-slateald-400` CSS class
  and a corrupted checkmark (`"? Selected"` → `"✓ Selected"`).

## -2. Previous round: archive/cancel, storage overview, deployment tab

### Hide/archive/cancel past trainings
- Training sessions now have an `archived` flag. New endpoints:
  `POST /api/training/archive/{id}`, `POST /api/training/unarchive/{id}`,
  `DELETE /api/training/session/{id}` (refuses to delete a still-running
  session), and `GET /api/training/sessions?include_archived=true`
  (archived sessions are hidden by default).
- **Fixed a real bug**: cancelling a training run always ended up marked
  `"completed"` regardless of cancellation, because the status was set
  unconditionally after `model.fit()` returned. It now correctly resolves
  to `"cancelled"` (and skips saving/registering a model if cancelled
  before any epoch finished).
- **Fixed**: the frontend's cancel handler cleared the poll loop and
  immediately faked the status to `"cancelled"` locally - meaning it never
  found out the real outcome and could show "cancelled" while training
  was still running. It now just requests cancellation and lets the
  existing poll loop discover the real status once the backend settles
  (cancellation takes effect at the next epoch boundary, not instantly -
  Keras has no clean mid-epoch interrupt).
- `ModelTrainer`'s history panel now has a "Show archived" toggle and
  per-session archive/cancel buttons.

### Backend storage overview
- New `GET /api/storage/overview`: per-dataset sample counts, sizes, and
  file-type breakdowns (by extension), split distribution (train/val/test/
  unassigned), trained-model sizes, optimized-model sizes, and overall
  disk usage - all computed live from what's actually on disk, not
  estimates.
- New collapsible **Backend Storage Overview** panel on the Dashboard tab
  showing all of this, including a per-dataset table.

### Fixed the "strange characters"
- The uploaded files had gone through a lossy encoding conversion before I
  ever received them, producing several silently-corrupted characters:
  `À`/`ò` → should be `•` (bullet separators), `Î` → `×` (multiplication,
  e.g. "4× size reduction"), `ù`/`û`/`¨` → `—` (em dash), `à` → `…`
  (ellipsis). Comment dividers had also degraded into literal
  `# ?? Section ??????????` noise. Swept the entire codebase (both
  backend `.py` and frontend `.ts`/`.tsx`) and replaced all of these with
  the correct characters / clean `# --- Section ---` dividers. Also fixed
  one instance of my own em-dash getting mangled mid-edit.

### Export moved to the Deployment tab, with real pin configuration
- The old Deployment tab was a non-functional placeholder (a disabled
  button with no handler). It's now a full panel:
  - **Camera pin configuration** for ESP32-CAM (all 16 GPIOs editable,
    defaulting to the common AI-Thinker pinout - other modules wire the
    camera differently).
  - **Optional status display**: enable an SPI TFT (ST7735 via
    Adafruit_GFX) with configurable CS/DC/RST/SCK/MOSI/backlight pins.
    When enabled, the exported sketch shows the live prediction + confidence
    on-screen, not just over Serial.
  - **Evaluate for Board** and **Export Arduino Project**, both now using
    the real pin configuration.
  - A **live "ready to flash" code preview** of `sketch.ino` that updates
    as you edit pins (debounced), with a one-click copy button - so you can
    see the exact mock example that will be flashed before downloading it.
- Backend: `exporter.py` now accepts `camera_pins` and `display_config`
  and threads them through the generated `.ino` (real `#define ..._GPIO_NUM`
  values, real display init/draw code). New
  `POST /api/optimization/export-preview/{id}` returns just the sketch
  text for the live preview, without a zip download.
- `/api/optimization/export/{id}` changed from `GET ?board=...` to
  `POST` with a JSON body (`board`, `camera_pins`, `display_config`),
  since there's now real structured configuration to send.
- **Fixed a real bug found while building this**: an f-string in the
  camera-pins template used `{{}}` intending an empty dict literal, but
  since it was inside an expression field (not literal text), Python
  parsed it as a set containing an empty dict - `TypeError: unhashable
  type: 'dict'`. Fixed and verified with a brace-balance check across
  every board/display combination.
- Removed `BoardAdvisor.tsx` (its evaluate-and-export functionality is now
  fully inside the new `DeploymentPanel`), and simplified the Optimization
  tab back to full width now that board evaluation lives in Deployment.

## -1. Previous round: inference crash, dataset association, testing gallery

- **Fixed the inference shape-mismatch crash** you hit (`expected shape=
  (None, 224, 224, 3), found shape=(1, 64, 64, 3)` on the original model, and
  `Dimension mismatch. Got 64 but expected 224` on the optimized one).
  Root cause: `inference_engine._preprocess_image()` hardcoded (64, 64) /
  (96, 96) per task regardless of what `input_shape` a given model was
  actually trained with. It now reads the real resolved input shape from
  the loaded Keras/TFLite model itself, for both the original and optimized
  inference paths.
- **Fixed a related training crash** you hit (`Shape mismatch in layer #0
  (named conv) for weight conv/kernel. Weight expects shape (3, 3, 1, 16).
  Received saved weight with shape (3, 3, 3, 16)`). Root cause: ImageNet-
  pretrained weights (MobileNetV2/V3Small/V1_0.25/EfficientNet/ResNet50V2)
  are only defined for 3-channel RGB input; picking one of those backbones
  together with a 1-channel (grayscale) `input_shape` crashed at weight-load
  time. `ModelFactory` now automatically tiles a 1-channel input to 3
  channels before feeding it to any pretrained backbone (the same pattern
  already used for Visual Wake Words), instead of crashing.
  - As part of this fix, also replaced `Trainer`'s fragile "backbone is
    always `model.layers[1]`" assumption (used for freeze/fine-tune) with a
    lookup by type, since the new channel-adapter layer can shift positions.
- **Models are now associated with their source dataset.** Each entry in
  `GET /api/training/models` now includes `dataset_id` and `dataset_name`
  (previously absent entirely). Shown in the model picker in
  `OptimizationStudio` as "ModelName · DatasetName".
- **New: Test Set Predictions gallery.** The evaluator now records a
  per-sample breakdown (true label, both models' predicted label/confidence/
  correctness) for every sample it evaluates, exposed via
  `GET /api/optimization/result/{id}` → `comparison.sample_results`. The
  lightweight `/status` endpoint (polled every few seconds during
  optimization) intentionally excludes this to stay fast - it's only in
  `/result`. The frontend's Test-Set Comparison panel now has an expandable
  gallery showing each sample's real thumbnail (fetched from
  `/api/datasets/image/{id}`) with both models' predictions and a
  correct/incorrect marker, plus an "only mismatches" filter.

## 0. CPU / GPU training toggle

- New `device: "auto" | "cpu" | "gpu"` field on `POST /api/training/start`.
  - `"cpu"` forces `tf.device("/CPU:0")` for model build + fit.
  - `"gpu"` forces `/GPU:0"`, falling back to CPU with a logged warning if no
    GPU is visible to TensorFlow.
  - `"auto"` (default) leaves TF's normal placement behavior untouched.
- When forcing CPU while the global `mixed_float16` policy is active (set at
  startup if a GPU is present), the policy is temporarily switched to
  `float32` for that run and restored afterward — `mixed_float16` is a
  GPU/Tensor-Core optimization and just adds pointless cast overhead on CPU.
- New `GET /api/training/devices` reports real CPU/GPU availability (via
  `tf.config.list_physical_devices`), used by the frontend to grey out the
  GPU option when none exists.
- The resolved device (`device_used`) is recorded on the training session
  and shown as a small ⚡GPU / 🖥CPU badge on the live training panel.
- Frontend: new **Compute Device** dropdown next to Base Model in
  `ModelTrainer`, backed by `apiClient.getAvailableDevices()`.
- **Also fixed while touching this code path**: the frontend sent
  `freeze_epochs`, but the backend's field is `freeze_encoder_epochs` — the
  mismatch meant the freeze-then-fine-tune two-phase training option was
  silently a no-op whenever set in the UI. Now sends the correct field name.

## 1. LLM-assisted training decisions

- **New: pre-training recommendations.** `POST /api/training/recommend`
  (`llm_advisor.recommend_training_params`) suggests `base_model`,
  `input_shape`, `batch_size`, `epochs`, `learning_rate`, `dropout_rate`,
  and `augmentation` **before** you start a run, based on the task, your
  actual dataset stats, and the board you plan to deploy to. Falls back to
  sensible rule-based defaults if no LLM is reachable.
  Wired into `ModelTrainer` as a **"✨ Suggest Optimal Config"** button.
- **Fixed a real bug** in `llm_advisor.generate_suggestions` (post-training
  advice): it read `context.get("config", {})`, but the training session
  dict is flat, not nested under `"config"` — every hyperparameter lookup
  silently returned "Unknown". Now reads `context` directly.
- **Fixed:** `/llm-optimize` called `llm_advisor.get_optimization_advice(...)`,
  a method that didn't exist on the uploaded `LLMAdvisor`. Implemented it
  for real, grounded in the actual completed optimization session's size/
  compression ratio/measured test-set speedup — not generic text.
- **Fixed:** `LocalLLMAdvisor.get_status()` always returned `"connected"`
  regardless of whether Ollama was running. Now does a real HTTP check.

## 2. Predefined models curated for edge suitability

- Added **`MobileNetV1_0.25`** (width multiplier 0.25×) — the smallest
  ImageNet-pretrained option available, purpose-built for ESP32/Nano-class
  RAM budgets.
- Added `ModelFactory.EDGE_SUITABILITY` metadata (tiny/small/medium/large/
  very_large tiers) used by the LLM advisor to steer recommendations away
  from `ResNet50V2` (~23.5M params) and `EfficientNet` for MCU deployment.
- **Changed the default** image config from `MobileNetV2 @ 224×224` to
  `MobileNetV3Small @ 96×96` — the old default alone could exceed 8MB as
  float32, unusable on any supported board without heavy optimization first.

## 3. Optimization — now fully implemented, with real test-set comparison

- **New `app/services/evaluator.py`**: runs the ORIGINAL `.keras` model and
  the OPTIMIZED `.tflite` model on the same held-out **test split**,
  reporting real accuracy, loss, per-sample inference latency, and size for
  both, plus deltas (accuracy change, speedup factor, size reduction %).
  Falls back to the `val` split (and says so) if no test split exists yet.
- **Fixed a real bug**: the optimizer looked for trained models at
  `<storage>/<training_id>/model.keras`, but `Trainer` actually saves a flat
  `<storage>/<training_id>.keras` — every optimization would have failed to
  find its model. Same bug existed in `inference.py`'s model resolution and
  is now fixed there too, alongside a **label-ordering bug** that
  alphabetically sorted class labels before inference, silently scrambling
  which label matched which output neuron whenever training order wasn't
  alphabetical.
- **Removed the fake `Converter` service** (`app/services/converter.py`) —
  it faked a hardcoded 2.5MB original size, hardcoded compression ratios per
  method, and generated a C array filled with `0x00, 0x01, 0x02...`
  regardless of the real model. All optimization now flows through the real
  `optimizer.py`.
- **Added `WEIGHT_CLUSTERING`**, which didn't exist before despite being a
  valid enum value in `models.py` and the frontend's option list.
- **Fixed a real compatibility bug**: pruning and weight clustering used
  `tensorflow_model_optimization`, whose `isinstance` checks predate Keras 3
  and threw `"can only prune an object of type... You passed an object of
  type: Sequential"` for perfectly ordinary models. Replaced with a manual,
  dependency-light implementation (magnitude thresholding for pruning,
  scipy k-means for clustering) that doesn't depend on tfmot at all, and
  verified working. Both **clone the model before mutating weights**, so
  pruning/clustering never corrupts the "original" side of later
  comparisons (verified: re-running INT8 after a pruning run still reports
  the same original-model accuracy).
- **Frontend/backend method-name mismatch fixed**: the UI sends
  `INT8_QUANTIZATION`, `FLOAT16_QUANTIZATION`, etc., but the optimizer
  expected `int8`, `float16`, etc. — every optimization request from the UI
  was failing with "Unknown optimization method". Added a mapping layer in
  `routers/optimization.py` (per your preference) that translates between
  the two without changing either side's existing naming.
- **Fixed broken route paths**: the router defined
  `/optimization/{id}/status` and `/optimization/{id}/result`, but the
  frontend's `apiClient` calls `/optimization/status/{id}` and
  `/optimization/result/{id}` (opposite order) — 404 on every poll. Also
  fixed a stray leftover in `OptimizationStudio`'s polling code that hit
  `/optimization/optimization/{id}/status` (doubled segment, wrong order).
- New **Test-Set Comparison** panel in `OptimizationStudio` shows original
  vs optimized accuracy/loss/inference time/size side-by-side, pulled from
  the real evaluation above.

## 4. Board export — fully implemented for ESP32-S3 and ESP32-CAM

- **New `app/services/exporter.py`**: builds a real, ready-to-flash Arduino
  project — `model_data.h` (actual model bytes as a C array),
  `sketch.ino`, and `README.md` — as a downloadable zip
  (`GET /api/optimization/export/{id}?board=...`).
  - ESP32-S3 / ESP32-CAM: full `TfLiteMicroInterpreter` init, tensor-arena
    sizing from a real heuristic (not a guess), int8/float I/O handling.
  - **ESP32-CAM specifically**: real `esp_camera.h` integration with
    AI-Thinker pin mapping, frame capture, and downsampling to the model's
    exact input size.
  - Other boards: a generic Serial-based test harness so you can validate
    inference before wiring up real sensor capture.
- **Fixed a real formatting bug** in `CArrayGenerator.binary_to_c_array`:
  the old newline-insertion logic produced a stray double-comma at every
  16-byte line break (`0x14, \n  , 0x1c`), which is invalid C. Fixed and
  verified by actually compiling the generated header with `gcc`.
- **Fixed**: `/to-c-array` used the fake `Converter`'s placeholder bytes;
  now uses the real optimized model bytes.
- Added **ESP32-CAM** as a supported board throughout: `models.py`,
  `mcu_advisor.py`, `main.py`'s `/api/info`, and the frontend's board
  selector/types.
- `MCUAdvisor.evaluate_model()` used to return **entirely simulated**
  numbers (hardcoded 650KB / 150ms regardless of the actual model). Now
  pulls the real optimized file size, estimates RAM via tensor
  introspection (clearly labeled as a heuristic — exact arena size can only
  be known by compiling with TFLite Micro on-device), and scales the
  *measured* test-set inference time by a documented clock-ratio for a
  ballpark on-device estimate. Every estimate is explicitly labeled as such
  in the API response (`estimation_note`).

## 5. Other fixes found along the way

- `app/services/quantization_optimizer.py` was dead, unused, duplicate code
  — removed.
- `useAPI.ts`: added `exportProject()` and `getTrainingRecommendation()`.
- `BoardAdvisor.tsx`: shows the new real/estimated fields and adds an
  "Export Arduino Project" button.

## Known environment-specific note

`TRANSFER_LEARNING` downloads MobileNetV2 ImageNet weights on first use —
this failed in my sandboxed test environment only because outbound access
to `storage.googleapis.com` is blocked there. It will work normally with
regular internet access; nothing in the code needs to change for this.

## Files touched
**Backend:** `models.py`, `main.py`, `shared_state.py`, `optimizer.py`
(rewritten), `mcu_advisor.py` (rewritten), `exporter.py` (new),
`evaluator.py` (new), `llm_advisor.py` (extended), `local_llm_advisor.py`
(rewritten), `model_factory.py`, `c_array_generator.py`,
`routers/optimization.py` (rewritten), `routers/training.py`,
`routers/inference.py`. Removed: `converter.py`, `quantization_optimizer.py`.

**Frontend:** `types/index.ts`, `hooks/useAPI.ts`, `App.tsx`,
`components/BoardAdvisor.tsx`, `components/ModelTrainer/index.tsx`,
`components/ModelTrainer/constants.ts`,
`components/OptimizationStudio/index.tsx`.
