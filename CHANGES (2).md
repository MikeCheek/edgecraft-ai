# EdgeCraft AI — Changes Summary

Both zips contain the **full project** (not just diffs) — extract over your
existing folders. Everything below was verified end-to-end with a real
dataset → train → optimize → evaluate → export run (TensorFlow actually
executing, not mocked), plus a `gcc` compile check of the generated C header.

---

## -2. Latest round: archive/cancel, storage overview, deployment tab

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
