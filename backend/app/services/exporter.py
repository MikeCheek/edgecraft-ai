"""
exporter.py
-----------
Turns a completed optimization session into a real, ready-to-open Arduino
project: model_data.h (actual model bytes as a C array), a full .ino sketch
wired for the chosen board AND the user's actual hardware pin configuration
(camera pins for ESP32-CAM, optional attached display pins), and a README
with setup instructions.

This replaces the previous fake/disconnected export path (the old
Converter.generate_c_array produced a placeholder 0x00,0x01,0x02... array
unrelated to the real model).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


from app.utils.c_array_generator import CArrayGenerator


# ---------------------------------------------------------------------------
# Default pin configurations (used when the user doesn't override them)
# ---------------------------------------------------------------------------

# AI-Thinker ESP32-CAM default camera pinout - by far the most common
# ESP32-CAM module, so these are sane defaults, but every field is
# user-overridable since other ESP32-CAM variants (M5Stack, TTGO, etc.)
# wire the camera to different GPIOs.
DEFAULT_CAMERA_PINS: Dict[str, int] = {
    "pwdn": 32, "reset": -1, "xclk": 0, "siod": 26, "sioc": 27,
    "y9": 35, "y8": 34, "y7": 39, "y6": 36, "y5": 21, "y4": 19, "y3": 18, "y2": 5,
    "vsync": 25, "href": 23, "pclk": 22,
}

# A generic SPI TFT (ST7735-family) wiring - common on ESP32 dev boards
# breadboarded for a small status display. Fully user-overridable.
DEFAULT_DISPLAY_PINS: Dict[str, Any] = {
    "cs": 15, "dc": 2, "rst": 4, "sck": 18, "mosi": 23, "backlight": None,
}


# ---------------------------------------------------------------------------
# Context lookup
# ---------------------------------------------------------------------------

def _get_export_context(optimization_id: str) -> Dict[str, Any]:
    from app.services.optimizer import get_session, get_output_path, _get_training_context

    session = get_session(optimization_id)
    if not session:
        raise ValueError(f"Optimization session {optimization_id} not found")
    if session.get("status") != "completed":
        raise ValueError("Optimization is not completed yet - nothing to export.")

    output_path = get_output_path(optimization_id)
    if not output_path or not output_path.exists():
        raise FileNotFoundError("Optimized .tflite file not found on disk.")

    tflite_bytes = output_path.read_bytes()
    ctx = _get_training_context(session["training_id"])

    return {
        "session": session,
        "tflite_bytes": tflite_bytes,
        "task": ctx.get("task") or "IMAGE_CLASSIFICATION",
        "input_shape": ctx.get("input_shape") or (96, 96, 1),
        "labels": ctx.get("labels") or [],
        "method": session.get("frontend_method", session.get("method")),
    }


# ---------------------------------------------------------------------------
# C array (used both standalone by /to-c-array and inside the full package)
# ---------------------------------------------------------------------------

def generate_c_array_only(optimization_id: str) -> str:
    """Real C header string built from the actual optimized model bytes."""
    ctx = _get_export_context(optimization_id)
    return CArrayGenerator.binary_to_c_array(ctx["tflite_bytes"], model_name="model")


# ---------------------------------------------------------------------------
# Tensor arena sizing (reuse the mcu_advisor heuristic for consistency)
# ---------------------------------------------------------------------------

def _estimate_arena_bytes(optimization_id: str, board: str) -> int:
    from app.services.mcu_advisor import MCUAdvisor

    advisor = MCUAdvisor()
    result = advisor.evaluate_model(optimization_id, board)
    ram_kb = result.get("ram_usage_kb", 64)
    # Round up to the nearest 4KB and add 20% headroom - safer to allocate
    # slightly more arena than the heuristic predicts than to crash from
    # kTfLiteError "AllocateTensors failed" on-device.
    arena_bytes = int((ram_kb * 1024) * 1.2)
    arena_bytes = ((arena_bytes // 4096) + 1) * 4096
    return max(arena_bytes, 20 * 1024)


# ---------------------------------------------------------------------------
# Sketch templates
# ---------------------------------------------------------------------------

_TASK_OUTPUT_IS_BINARY = {"VISUAL_WAKE_WORDS"}
_AUDIO_TASKS = {"AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING"}


def _labels_array_cpp(labels: List[str]) -> str:
    if not labels:
        labels = ["class_0", "class_1"]
    quoted = ", ".join(f'"{l}"' for l in labels)
    return f"const char* kLabels[] = {{{quoted}}};\nconst int kNumLabels = {len(labels)};"


def _op_resolver_block() -> str:
    return (
        "  // Using AllOpsResolver for maximum compatibility with any exported\n"
        "  // model. Once your model architecture is finalised, you can switch to\n"
        "  // tflite::MicroMutableOpResolver<N> and register only the ops your\n"
        "  // model actually needs, which meaningfully reduces flash usage.\n"
        "  static tflite::AllOpsResolver resolver;"
    )


def _display_includes_and_globals(display_config: Optional[Dict[str, Any]]) -> str:
    """Adafruit_ST7735 + Adafruit_GFX is the most common cheap SPI TFT combo
    used in ESP32 TinyML demos, so that's what this generates when a display
    is enabled. Pins are fully user-configured (set in the Deployment tab)."""
    if not display_config or not display_config.get("enabled"):
        return ""

    pins = {**DEFAULT_DISPLAY_PINS, **(display_config or {})}
    backlight_define = f"#define TFT_BL   {pins['backlight']}" if pins.get("backlight") is not None else ""

    return f"""
// ---------------------------------------------------------------------------
// Status display (SPI TFT, ST7735 driver via Adafruit_GFX)
// ---------------------------------------------------------------------------
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

#define TFT_CS   {pins['cs']}
#define TFT_DC   {pins['dc']}
#define TFT_RST  {pins['rst']}
#define TFT_SCK  {pins['sck']}
#define TFT_MOSI {pins['mosi']}
{backlight_define}

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);

void initDisplay() {{
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(0, 0);
  tft.println("EdgeCraft AI");
  tft.println("Waiting for model...");
}}

void showPredictionOnDisplay(const char* label, float confidence) {{
  tft.fillScreen(ST77XX_BLACK);
  tft.setCursor(0, 0);
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_GREEN);
  tft.println(label);
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(0, 40);
  tft.print("Confidence: ");
  tft.print(confidence * 100.0f, 1);
  tft.println("%");
}}
"""


def _base_ino_header(
    model_name: str, task: str, board: str, input_shape: Tuple[int, ...],
    labels: List[str], arena_bytes: int, display_config: Optional[Dict[str, Any]],
) -> str:
    h, w = (input_shape[0], input_shape[1]) if len(input_shape) >= 2 else (96, 96)
    c = input_shape[2] if len(input_shape) >= 3 else 1
    is_binary = task in _TASK_OUTPUT_IS_BINARY
    display_enabled = bool(display_config and display_config.get("enabled"))

    display_call = (
        "    showPredictionOnDisplay(kIsBinaryOutput ? (readOutput(0) >= 0.5f ? kLabels[1 % kNumLabels] : kLabels[0]) : kLabels[best_idx], best_val);\n"
        if display_enabled else ""
    )

    return f"""\
// {model_name} - generated by EdgeCraft AI
// Task: {task}
// Target board: {board}
// Input shape: {list(input_shape)}
//
// Required Arduino libraries (install via Library Manager):
//   - "Chirale_TensorFlowLite" (or "TensorFlowLite_ESP32" depending on core version)
//   - ESP32 board package (esp32 by Espressif Systems) >= 2.0.x
{"//   - \"Adafruit GFX Library\" and \"Adafruit ST7735 and ST7789 Library\" (for the status display)" if display_enabled else ""}

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
{_display_includes_and_globals(display_config)}
{_labels_array_cpp(labels)}

namespace {{
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

constexpr int kTensorArenaSize = {arena_bytes};  // bytes; heuristic estimate + 20% headroom
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}}  // namespace

const int kInputHeight = {h};
const int kInputWidth  = {w};
const int kInputChannels = {c};
const bool kIsBinaryOutput = {"true" if is_binary else "false"};

bool initializeModel() {{
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {{
    MicroPrintf("Model schema version mismatch: %d vs supported %d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }}

{_op_resolver_block()}

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {{
    MicroPrintf("AllocateTensors() failed - try increasing kTensorArenaSize");
    return false;
  }}

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  return true;
}}

// Quantise a float [0,1] input sample into the model's expected dtype,
// writing directly into the interpreter's input tensor at `index`.
void setInputSample(int index, float value_0_to_1) {{
  if (model_input->type == kTfLiteInt8) {{
    float scale = model_input->params.scale;
    int zero_point = model_input->params.zero_point;
    model_input->data.int8[index] = static_cast<int8_t>(value_0_to_1 / scale + zero_point);
  }} else {{
    model_input->data.f[index] = value_0_to_1;
  }}
}}

float readOutput(int index) {{
  if (model_output->type == kTfLiteInt8) {{
    float scale = model_output->params.scale;
    int zero_point = model_output->params.zero_point;
    return (model_output->data.int8[index] - zero_point) * scale;
  }}
  return model_output->data.f[index];
}}

void printPrediction() {{
  int num_outputs = kIsBinaryOutput ? 1 : kNumLabels;
  int best_idx = 0;
  float best_val = -1e9f;
  for (int i = 0; i < num_outputs; i++) {{
    float v = readOutput(i);
    if (v > best_val) {{ best_val = v; best_idx = i; }}
  }}

  if (kIsBinaryOutput) {{
    float p = readOutput(0);
    Serial.print("Prediction: ");
    Serial.println(p >= 0.5f ? kLabels[1 % kNumLabels] : kLabels[0]);
    Serial.print("Confidence: "); Serial.println(p, 4);
  }} else {{
    Serial.print("Prediction: ");
    Serial.println(kLabels[best_idx]);
    Serial.print("Confidence: "); Serial.println(best_val, 4);
  }}
{display_call}}}
"""


def _generic_setup_loop(is_audio: bool, display_config: Optional[Dict[str, Any]]) -> str:
    """For boards without a camera: reads raw float samples over Serial
    (one line of comma-separated values matching the model's flattened
    input size) and runs inference on them. Replace `readSensorFrame()`
    with your actual sensor/microphone capture code."""
    display_enabled = bool(display_config and display_config.get("enabled"))
    display_init = "  initDisplay();\n" if display_enabled else ""

    return f"""
// ---------------------------------------------------------------------------
// Board-specific input capture
// ---------------------------------------------------------------------------
// This board has no built-in camera. Replace readSensorFrame() with your
// actual {"microphone/MFCC" if is_audio else "sensor"} capture code. As a
// starting point this reads a comma-separated line of floats (already
// normalised to [0,1] the same way training data was) over Serial, so you
// can test end-to-end inference before wiring up real hardware capture.
bool readSensorFrame() {{
  if (!Serial.available()) return false;
  String line = Serial.readStringUntil('\\n');
  int idx = 0;
  int start = 0;
  int total = kInputHeight * kInputWidth * kInputChannels;
  for (int i = 0; i < line.length() && idx < total; i++) {{
    if (line[i] == ',' || i == line.length() - 1) {{
      String tok = line.substring(start, (i == line.length() - 1) ? i + 1 : i);
      setInputSample(idx++, tok.toFloat());
      start = i + 1;
    }}
  }}
  return idx == total;
}}

void setup() {{
  Serial.begin(115200);
  delay(1000);
  Serial.println("EdgeCraft AI - model init...");
{display_init}  if (!initializeModel()) {{
    Serial.println("ERROR: model init failed");
    while (1) delay(1000);
  }}
  Serial.println("Model ready. Send a comma-separated normalised input line to run inference.");
}}

void loop() {{
  if (readSensorFrame()) {{
    if (interpreter->Invoke() != kTfLiteOk) {{
      Serial.println("ERROR: inference failed");
      return;
    }}
    printPrediction();
  }}
}}
"""


def _camera_pins_defines(camera_pins: Dict[str, int]) -> str:
    p = {**DEFAULT_CAMERA_PINS, **(camera_pins or {})}
    return (
        f"#define PWDN_GPIO_NUM     {p['pwdn']}\n"
        f"#define RESET_GPIO_NUM    {p['reset']}\n"
        f"#define XCLK_GPIO_NUM     {p['xclk']}\n"
        f"#define SIOD_GPIO_NUM     {p['siod']}\n"
        f"#define SIOC_GPIO_NUM     {p['sioc']}\n"
        f"#define Y9_GPIO_NUM       {p['y9']}\n"
        f"#define Y8_GPIO_NUM       {p['y8']}\n"
        f"#define Y7_GPIO_NUM       {p['y7']}\n"
        f"#define Y6_GPIO_NUM       {p['y6']}\n"
        f"#define Y5_GPIO_NUM       {p['y5']}\n"
        f"#define Y4_GPIO_NUM       {p['y4']}\n"
        f"#define Y3_GPIO_NUM       {p['y3']}\n"
        f"#define Y2_GPIO_NUM       {p['y2']}\n"
        f"#define VSYNC_GPIO_NUM    {p['vsync']}\n"
        f"#define HREF_GPIO_NUM     {p['href']}\n"
        f"#define PCLK_GPIO_NUM     {p['pclk']}"
    )


def _camera_setup_loop(
    input_shape: Tuple[int, ...], camera_pins: Optional[Dict[str, int]],
    display_config: Optional[Dict[str, Any]],
) -> str:
    h, w = input_shape[0], input_shape[1]
    channels = input_shape[2] if len(input_shape) >= 3 else 1
    pixfmt = "PIXFORMAT_GRAYSCALE" if channels == 1 else "PIXFORMAT_RGB565"
    display_enabled = bool(display_config and display_config.get("enabled"))
    display_init = "  initDisplay();\n" if display_enabled else ""
    display_call = (
        "  showPredictionOnDisplay(kIsBinaryOutput ? (readOutput(0) >= 0.5f ? kLabels[1 % kNumLabels] : kLabels[0]) : kLabels[0], 0.0f);\n"
        if display_enabled else ""
    )

    return f"""
// ---------------------------------------------------------------------------
// ESP32-CAM camera capture (pins configured in the Deployment tab)
// ---------------------------------------------------------------------------
#include "esp_camera.h"

{_camera_pins_defines(camera_pins or {})}

bool initCamera() {{
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  // Grab the smallest available frame size and crop/downsample in software
  // to the model's exact {w}x{h} input - avoids a JPEG decode step entirely.
  config.frame_size = FRAMESIZE_QQVGA;   // 160x120
  config.pixel_format = {pixfmt};
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {{
    Serial.println("ERROR: camera init failed");
    return false;
  }}
  return true;
}}

// Nearest-neighbour downsample from the captured frame to the model's
// {w}x{h}x{channels} input, normalising pixel values to [0,1].
void frameToModelInput(camera_fb_t* fb) {{
  int src_w = fb->width;
  int src_h = fb->height;
  int idx = 0;
  for (int y = 0; y < {h}; y++) {{
    int sy = y * src_h / {h};
    for (int x = 0; x < {w}; x++) {{
      int sx = x * src_w / {w};
      uint8_t pixel = fb->buf[sy * src_w + sx];  // grayscale byte-per-pixel
      setInputSample(idx++, pixel / 255.0f);
    }}
  }}
}}

void setup() {{
  Serial.begin(115200);
  delay(1000);
  Serial.println("EdgeCraft AI - camera + model init...");
{display_init}  if (!initCamera()) {{ while (1) delay(1000); }}
  if (!initializeModel()) {{
    Serial.println("ERROR: model init failed");
    while (1) delay(1000);
  }}
  Serial.println("Camera + model ready.");
}}

void loop() {{
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {{
    Serial.println("ERROR: camera capture failed");
    delay(500);
    return;
  }}

  frameToModelInput(fb);
  esp_camera_fb_return(fb);

  if (interpreter->Invoke() != kTfLiteOk) {{
    Serial.println("ERROR: inference failed");
    return;
  }}
  printPrediction();
{display_call}  delay(200);
}}
"""


def generate_arduino_sketch(
    model_name: str, task: str, board: str, input_shape: Tuple[int, ...],
    labels: List[str], arena_bytes: int,
    camera_pins: Optional[Dict[str, int]] = None,
    display_config: Optional[Dict[str, Any]] = None,
) -> str:
    header = _base_ino_header(model_name, task, board, input_shape, labels, arena_bytes, display_config)
    if board == "ESP32_CAM":
        body = _camera_setup_loop(input_shape, camera_pins, display_config)
    else:
        body = _generic_setup_loop(is_audio=task in _AUDIO_TASKS, display_config=display_config)
    return header + body


def _generate_readme(
    board: str, task: str, method: str, arena_bytes: int,
    input_shape: Tuple[int, ...], labels: List[str],
    camera_pins: Optional[Dict[str, int]], display_config: Optional[Dict[str, Any]],
) -> str:
    display_enabled = bool(display_config and display_config.get("enabled"))

    camera_note = (
        "\nThis board has an onboard camera. The sketch captures frames directly\n"
        "and feeds them to the model using the pins you configured in the\n"
        "Deployment tab (see the `#define ..._GPIO_NUM` block at the top of\n"
        "`sketch.ino` if you need to double check them against your module).\n"
        if board == "ESP32_CAM"
        else (
            "\nThis board has no built-in sensor for this task. The sketch reads a\n"
            "test input as a comma-separated line over Serial so you can validate\n"
            "inference before wiring up your real sensor/microphone capture code\n"
            "in `readSensorFrame()`.\n"
        )
    )

    display_section = ""
    if display_enabled:
        pins = {**DEFAULT_DISPLAY_PINS, **(display_config or {})}
        display_section = f"""
## Status display
A ST7735-driven SPI TFT is wired in, using these pins (as configured in the
Deployment tab):
- CS:   {pins['cs']}
- DC:   {pins['dc']}
- RST:  {pins['rst']}
- SCK:  {pins['sck']}
- MOSI: {pins['mosi']}
{f"- Backlight: {pins['backlight']}" if pins.get('backlight') is not None else "- Backlight: tied directly to 3.3V (no GPIO control configured)"}

Install **Adafruit GFX Library** and **Adafruit ST7735 and ST7789 Library**
via the Arduino Library Manager before flashing.
"""

    return f"""# EdgeCraft AI - Exported Model Package

## Contents
- `model_data.h` - the optimized model, embedded as a C byte array
- `sketch.ino`   - a ready-to-flash Arduino sketch for **{board}**

## Model info
- Task: {task}
- Optimization applied: {method}
- Input shape: {list(input_shape)}
- Classes: {", ".join(labels) if labels else "(none recorded)"}
- Estimated tensor arena: {arena_bytes // 1024} KB (heuristic + 20% headroom -
  if you see `AllocateTensors() failed` at runtime, increase `kTensorArenaSize`
  in `sketch.ino` and re-flash)

## Required Arduino libraries
1. Board package: **esp32** by Espressif Systems (Boards Manager) - version 2.0.x or newer
2. Library: **Chirale_TensorFlowLite** (or **TensorFlowLite_ESP32**, depending on
   what's available for your installed core version) - Library Manager
{"3. Library: built-in **esp32-camera** driver (bundled with the esp32 board package)" if board == "ESP32_CAM" else ""}
{"4. Libraries: **Adafruit GFX Library** + **Adafruit ST7735 and ST7789 Library**" if display_enabled else ""}

## Setup
1. Install the libraries above via the Arduino IDE Library Manager / Boards Manager.
2. Open `sketch.ino` and `model_data.h` in the same folder (Arduino requires the
   .ino filename to match the folder name - rename the folder if needed).
3. Select the correct board under Tools > Board (e.g. "ESP32S3 Dev Module" or
   "AI Thinker ESP32-CAM").
4. Set Tools > Partition Scheme to a scheme with enough app storage for the
   TensorFlow Lite Micro runtime (e.g. "Huge APP").
5. Flash the sketch.
{camera_note}{display_section}
## Testing
Open the Serial Monitor at 115200 baud after flashing. You should see
"Model ready" (and "Camera + model ready" on ESP32-CAM), followed by a
`Prediction: <label>` / `Confidence: <value>` line each time inference runs.
{"The status display will also show the current prediction." if display_enabled else ""}

## Notes on the tensor arena estimate
The RAM figure above comes from introspecting the exported TFLite model's
tensors on this PC, not from compiling it with TFLite Micro on the actual
device - the real allocator may need a different amount. If allocation
fails on-device, increase `kTensorArenaSize` in steps of a few KB and
re-flash until `AllocateTensors()` succeeds.
"""


# ---------------------------------------------------------------------------
# Full package (zip)
# ---------------------------------------------------------------------------

def generate_export_package(
    optimization_id: str, board: str,
    camera_pins: Optional[Dict[str, int]] = None,
    display_config: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Build a zip containing model_data.h, sketch.ino, and README.md,
    wired for the user's actual pin configuration (camera pins for
    ESP32-CAM, an optional attached status display for any board)."""
    ctx = _get_export_context(optimization_id)

    c_header = CArrayGenerator.binary_to_c_array(ctx["tflite_bytes"], model_name="model")
    arena_bytes = _estimate_arena_bytes(optimization_id, board)

    sketch = generate_arduino_sketch(
        model_name="model",
        task=ctx["task"],
        board=board,
        input_shape=ctx["input_shape"],
        labels=ctx["labels"],
        arena_bytes=arena_bytes,
        camera_pins=camera_pins,
        display_config=display_config,
    )

    readme = _generate_readme(
        board=board,
        task=ctx["task"],
        method=ctx["method"],
        arena_bytes=arena_bytes,
        input_shape=ctx["input_shape"],
        labels=ctx["labels"],
        camera_pins=camera_pins,
        display_config=display_config,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model_data.h", c_header)
        zf.writestr("sketch.ino", sketch)
        zf.writestr("README.md", readme)
    buf.seek(0)
    return buf.read()


def preview_sketch(
    optimization_id: str, board: str,
    camera_pins: Optional[Dict[str, int]] = None,
    display_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Return just the .ino text (no zip) so the frontend can show a live
    'ready to flash' code preview as the user edits pin values, without
    triggering a download each time."""
    ctx = _get_export_context(optimization_id)
    arena_bytes = _estimate_arena_bytes(optimization_id, board)
    return generate_arduino_sketch(
        model_name="model",
        task=ctx["task"],
        board=board,
        input_shape=ctx["input_shape"],
        labels=ctx["labels"],
        arena_bytes=arena_bytes,
        camera_pins=camera_pins,
        display_config=display_config,
    )
