"""
Turns a completed optimization session into a real, ready-to-open Arduino
project: model_data.h (actual model bytes as a C array), a full .ino sketch
wired for the chosen board AND the user's actual hardware configuration, and
a README with setup instructions.

Hardware configuration is now two independent, named choices rather than a
handful of raw pin dicts the caller had to know the shape of:

- `camera_config.module_preset` picks a known camera module (e.g. the
  AI-Thinker ESP32-CAM's OV2640, or an ESP32-S3-CAM board's OV3660) with its
  correct default pinout baked in. Any individual pin can still be
  overridden via `camera_config.pins_override` (or the legacy top-level
  `camera_pins` argument, kept for backward compatibility).
- `display_config.module_preset` picks a known display module + sketch
  style (e.g. a plain text status HUD, or a pixel-art bordered live-preview
  HUD like the one used on ESP32-CAM-class boards). Pins work the same way
  as before - set directly on `display_config` - but now default from the
  chosen preset instead of a single hardcoded default.

This replaces the previous fake/disconnected export path (the old
Converter.generate_c_array produced a placeholder 0x00,0x01,0x02... array
unrelated to the real model).
"""

from __future__ import annotations

import io
import zipfile
from typing import Dict, Any, List, Tuple, Optional

from app.services import preprocessing
from app.utils.c_array_generator import CArrayGenerator
from app.utils.mfcc_codegen import generate_mfcc_header

# ---------------------------------------------------------------------------
# Camera module presets
# ---------------------------------------------------------------------------
# Every preset is a full esp32-camera pinout. "module_type" is "integrated"
# (the camera ships soldered onto the board, e.g. ESP32-CAM/ESP32-S3-CAM) or
# "external" (a breakout wired by hand to an arbitrary dev board).

CAMERA_MODULE_PRESETS: Dict[str, Dict[str, Any]] = {
    "AI_THINKER_ESP32_CAM": {
        "label": "AI-Thinker ESP32-CAM (OV2640)",
        "module_type": "integrated",
        "pins": {
            "pwdn": 32, "reset": -1, "xclk": 0, "siod": 26, "sioc": 27,
            "y9": 35, "y8": 34, "y7": 39, "y6": 36, "y5": 21, "y4": 19, "y3": 18, "y2": 5,
            "vsync": 25, "href": 23, "pclk": 22,
        },
    },
    "ESP32_S3_CAM_OV3660": {
        "label": "ESP32-S3-CAM (OV3660)",
        "module_type": "integrated",
        "pins": {
            "pwdn": -1, "reset": -1, "xclk": 15, "siod": 4, "sioc": 5,
            "y9": 16, "y8": 17, "y7": 18, "y6": 12, "y5": 10, "y4": 8, "y3": 9, "y2": 11,
            "vsync": 6, "href": 7, "pclk": 13,
        },
    },
    "CUSTOM": {
        "label": "Custom / hand-wired camera",
        "module_type": "external",
        # A reasonable starting point (matches the old single hardcoded
        # default) - every pin is expected to be overridden for a real wiring.
        "pins": {
            "pwdn": 32, "reset": -1, "xclk": 0, "siod": 26, "sioc": 27,
            "y9": 35, "y8": 34, "y7": 39, "y6": 36, "y5": 21, "y4": 19, "y3": 18, "y2": 5,
            "vsync": 25, "href": 23, "pclk": 22,
        },
    },
}
DEFAULT_CAMERA_PRESET = "AI_THINKER_ESP32_CAM"

# Kept for any external code still importing the old name directly.
DEFAULT_CAMERA_PINS: Dict[str, int] = CAMERA_MODULE_PRESETS[DEFAULT_CAMERA_PRESET]["pins"]

# ---------------------------------------------------------------------------
# Display module presets
# ---------------------------------------------------------------------------
# "style" selects which sketch-generation code path is used:
#   - "simple"    - small ST7735 status HUD, text only (label + confidence).
#                   Works with or without a camera.
#   - "pixel_hud" - a bordered, pixel-art-styled 128x128 HUD with top/bottom
#                   status bars. When a camera is also enabled, it shows a
#                   live downsampled preview between the bars (matches the
#                   look of common ESP32-S3-CAM + ST7735 project sketches).
#                   Falls back to a text-only version of the same frame if
#                   no camera is enabled.

DISPLAY_MODULE_PRESETS: Dict[str, Dict[str, Any]] = {
    "NONE": {
        "label": "No display",
        "style": "none",
        "pins": {},
    },
    "ST7735_TEXT_HUD": {
        "label": "ST7735 SPI TFT - simple text HUD",
        "style": "simple",
        "pins": {"cs": 15, "dc": 2, "rst": 4, "sck": 18, "mosi": 23, "backlight": None},
    },
    "ST7735_PIXEL_HUD": {
        "label": "ST7735 128x128 SPI TFT - pixel-art bordered HUD (+ live camera preview if a camera is enabled)",
        "style": "pixel_hud",
        "pins": {"cs": 19, "dc": 45, "rst": 48, "sck": 47, "mosi": 21, "backlight": 46},
    },
}
DEFAULT_DISPLAY_PRESET = "ST7735_TEXT_HUD"

# Kept for any external code still importing the old name directly.
DEFAULT_DISPLAY_PINS: Dict[str, Any] = DISPLAY_MODULE_PRESETS[DEFAULT_DISPLAY_PRESET]["pins"]

MIC_MODULE_PRESETS: Dict[str, Dict[str, Any]] = {
    "NONE": {"label": "No microphone (stream audio over Serial)", "pins": {}},
    "INMP441_ESP32_S3": {"label": "INMP441 I2S mic on ESP32-S3", "pins": {"sck": 42, "ws": 41, "sd": 2}},
    "INMP441_ESP32": {"label": "INMP441 I2S mic on ESP32 / ESP32-CAM", "pins": {"sck": 14, "ws": 15, "sd": 13}},
}


def _resolve_mic(board: str, mic_config: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, int], str]:
    """(enabled, pins, preset_id). I2S capture is generated for ESP32 boards only."""
    mic_config = mic_config or {}
    preset_id = mic_config.get("module_preset") or ("INMP441_ESP32_S3" if board == "ESP32_S3_N16R8" else
                                                      "INMP441_ESP32" if board == "ESP32_CAM" else "NONE")
    preset = MIC_MODULE_PRESETS.get(preset_id, MIC_MODULE_PRESETS["NONE"])
    enabled = bool(mic_config.get("enabled", preset_id != "NONE")) and board.startswith("ESP32") and preset_id != "NONE"
    pins = dict(preset["pins"])
    pins.update({k: v for k, v in mic_config.items() if k in ("sck", "ws", "sd")})
    pins = _validate_pins(pins)
    if enabled and any(pins.get(k) is None for k in ("sck", "ws", "sd")):
        raise ValueError("Microphone needs sck, ws and sd pins.")
    return enabled, pins, preset_id


def list_hardware_presets() -> Dict[str, Any]:
    """Catalog consumed by the frontend to populate the camera/display
    module dropdowns, including default pins so a preview can be shown
    before the user customises anything."""
    return {
        "camera_modules": [
            {"id": k, "label": v["label"], "module_type": v["module_type"], "pins": v["pins"]}
            for k, v in CAMERA_MODULE_PRESETS.items()
        ],
        "display_modules": [
            {"id": k, "label": v["label"], "style": v["style"], "pins": v["pins"]}
            for k, v in DISPLAY_MODULE_PRESETS.items()
        ],
        "mic_modules": [
            {"id": k, "label": v["label"], "pins": v["pins"]}
            for k, v in MIC_MODULE_PRESETS.items()
        ],
    }

def _resolve_camera_pins(
    camera_config: Optional[Dict[str, Any]], legacy_camera_pins: Optional[Dict[str, int]]
) -> Tuple[Dict[str, int], str, str]:
    """Returns (pins, module_type, preset_id)."""
    camera_config = camera_config or {}
    preset_id = camera_config.get("module_preset", DEFAULT_CAMERA_PRESET)
    preset = CAMERA_MODULE_PRESETS.get(preset_id, CAMERA_MODULE_PRESETS[DEFAULT_CAMERA_PRESET])

    pins = dict(preset["pins"])
    pins.update(camera_config.get("pins_override") or {})
    # `legacy_camera_pins` is the original top-level `camera_pins` argument -
    # kept working exactly as before for any existing caller that doesn't
    # know about presets yet.
    pins.update(legacy_camera_pins or {})

    module_type = camera_config.get("module_type", preset["module_type"])
    return _validate_pins(pins), module_type, preset_id

def _resolve_display(
    display_config: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any], str, str]:
    """Returns (enabled, pins, style, preset_id)."""
    display_config = display_config or {}
    enabled = bool(display_config.get("enabled"))
    preset_id = display_config.get("module_preset", DEFAULT_DISPLAY_PRESET)
    preset = DISPLAY_MODULE_PRESETS.get(preset_id, DISPLAY_MODULE_PRESETS[DEFAULT_DISPLAY_PRESET])

    pins = dict(preset["pins"])
    # Manual pin keys have always lived directly on display_config alongside
    # "enabled" - keep that convention so existing overrides keep working.
    override_keys = {"cs", "dc", "rst", "sck", "mosi", "backlight"}
    pins.update({k: v for k, v in display_config.items() if k in override_keys})

    style = display_config.get("style", preset["style"])
    if style not in ("none", "simple", "pixel_hud"):
        style = preset["style"]
    return enabled, _validate_pins(pins), style, preset_id

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
    tflm = (session.get("metrics") or {}).get("tflm") or {}
    ops = tflm.get("ops")
    output_shape: List[int] = []
    try:
        from app.services.mcu_advisor import graph_ops
        from app.services.evaluator import make_interpreter

        if not ops:
            ops = sorted({o["op_name"] for o in graph_ops(tflite_bytes)})
        interp = make_interpreter(tflite_bytes, no_delegate=True)
        output_shape = [int(d) for d in interp.get_output_details()[0]["shape"]]
        input_shape = tuple(int(d) for d in interp.get_input_details()[0]["shape"][1:])
    except Exception:
        input_shape = tuple(ctx.get("input_shape") or (96, 96, 1))

    return {
        "session": session,
        "tflite_bytes": tflite_bytes,
        "task": ctx.get("task") or "IMAGE_CLASSIFICATION",
        "input_shape": input_shape,
        "labels": ctx.get("labels") or [],
        "method": session.get("frontend_method", session.get("method")),
        "ops": ops or [],
        "binary_output": bool(output_shape) and output_shape[-1] == 1 and len(output_shape) == 2,
        "tflm": tflm,
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
    # Measured TFLM arenas get 10% headroom (the device build can differ
    # slightly from the host one); estimates get 20%. Rounded up to 4 KB.
    headroom = 1.1 if result.get("ram_estimation_method") == "tflite_micro_measured" else 1.2
    arena_bytes = int((ram_kb * 1024) * headroom)
    arena_bytes = ((arena_bytes // 4096) + 1) * 4096
    return max(arena_bytes, 20 * 1024)

# ---------------------------------------------------------------------------
# Sketch templates
# ---------------------------------------------------------------------------

_AUDIO_TASKS = {"AUDIO_CLASSIFICATION", "KEYWORD_SPOTTING"}

def _c_string(text: str) -> str:
    """Escape arbitrary text for a C string literal."""
    out = []
    for ch in str(text):
        if ch in ('"', "\\"):
            out.append("\\" + ch)
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.extend(f"\\x{b:02x}\"\"" for b in ch.encode("utf-8"))
    return '"' + "".join(out) + '"'


def _labels_array_cpp(labels: List[str]) -> str:
    if not labels:
        labels = ["class_0", "class_1"]
    quoted = ", ".join(_c_string(l) for l in labels)
    return "const char* kLabels[] = {" + quoted + "};\nconst int kNumLabels = " + str(len(labels)) + ";"


# TFLite builtin op name -> MicroMutableOpResolver method. Most follow the
# CamelCase rule in _resolver_method; these are the exceptions.
_RESOLVER_EXCEPTIONS = {
    "BATCH_MATMUL": "AddBatchMatMul",
    "CUMSUM": "AddCumSum",
    "PADV2": "AddPadV2",
    "UNIDIRECTIONAL_SEQUENCE_LSTM": "AddUnidirectionalSequenceLSTM",
    "L2_NORMALIZATION": "AddL2Normalization",
    "L2_POOL_2D": "AddL2Pool2D",
    "LOGISTIC": "AddLogistic",
    "SVDF": "AddSvdf",
}


def _resolver_method(op: str) -> str:
    if op in _RESOLVER_EXCEPTIONS:
        return _RESOLVER_EXCEPTIONS[op]
    return "Add" + "".join(part if part[0].isdigit() else part.capitalize() for part in op.split("_"))


def _op_resolver_block(ops: Optional[List[str]] = None) -> str:
    # De-duplicated so the template capacity matches the Add*() calls (TFLM
    # rejects a second registration of the same op anyway).
    ops = list(dict.fromkeys(o for o in (ops or []) if o and o != "DELEGATE"))
    if not ops:
        return (
            "  // Op list unavailable for this model - registering every op.\n"
            "  static tflite::AllOpsResolver resolver;"
        )
    lines = [f"  // Only the {len(ops)} ops this model uses (saves flash vs AllOpsResolver).",
             f"  static tflite::MicroMutableOpResolver<{len(ops)}> resolver;"]
    lines += [f"  resolver.{_resolver_method(op)}();" for op in ops]
    return "\n".join(lines)


def _validate_pin(name: str, value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        pin = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Pin '{name}' must be an integer GPIO number, got {value!r}")
    if not -1 <= pin <= 60:
        raise ValueError(f"Pin '{name}' = {pin} is outside the valid GPIO range (-1..60)")
    return pin


def _validate_pins(pins: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _validate_pin(k, v) for k, v in pins.items()}


def _resolve_camera_enabled(board: str, camera_config: Optional[Dict[str, Any]]) -> bool:
    """ESP32-CAM (and any board whose camera preset is "integrated") has an
    onboard camera, so it's on by default unless explicitly disabled. Any
    other board has no camera unless the user explicitly opts in to wiring
    one up via GPIO (camera_config.enabled)."""
    camera_config = camera_config or {}
    if "enabled" in camera_config:
        return bool(camera_config["enabled"])
    preset_id = camera_config.get("module_preset")
    if preset_id and CAMERA_MODULE_PRESETS.get(preset_id, {}).get("module_type") == "integrated":
        return True
    return board == "ESP32_CAM"

# ---------------------------------------------------------------------------
# "simple" display style - small text-only status HUD
# ---------------------------------------------------------------------------

def _display_simple_includes_and_globals(
    pins: Dict[str, Any], camera_enabled: bool, input_shape: Tuple[int, ...],
) -> str:
    """Adafruit_ST7735 + Adafruit_GFX, driven over software SPI so any GPIO
    combination works (not just the board's dedicated hardware SPI pins).
    When a camera is ALSO enabled, this adds a second helper
    (showCameraFrameAndPrediction) that draws the live downsampled camera
    frame plus the prediction/confidence overlay - a basic live-preview UI,
    not just a status line."""
    backlight_define = f"#define TFT_BL   {pins['backlight']}" if pins.get("backlight") is not None else ""
    backlight_init = "  pinMode(TFT_BL, OUTPUT);\n  digitalWrite(TFT_BL, HIGH);\n" if pins.get("backlight") is not None else ""

    camera_preview_fn = ""
    if camera_enabled:
        camera_preview_fn = """
// Draws the live (downsampled, model-resolution) camera frame in the
// top-left corner, then the prediction + confidence below it - a basic
// live "camera streaming + inference" UI.
void showCameraFrameAndPrediction(const uint16_t* frame565, int frameW, int frameH,
                                   const char* label, float confidence) {
  tft.fillScreen(ST77XX_BLACK);
  tft.drawRGBBitmap(0, 0, frame565, frameW, frameH);
  tft.setCursor(0, frameH + 4);
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_GREEN);
  tft.println(label);
  tft.setTextColor(ST77XX_WHITE);
  tft.print("Confidence: ");
  tft.print(confidence * 100.0f, 1);
  tft.println("%");
}
"""

    return f"""
// ---------------------------------------------------------------------------
// Status display (SPI TFT, ST7735 driver via Adafruit_GFX, software SPI)
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

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCK, TFT_RST);

void initDisplay() {{
{backlight_init}  tft.initR(INITR_BLACKTAB);
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
{camera_preview_fn}"""

# ---------------------------------------------------------------------------
# "pixel_hud" display style - bordered pixel-art 128x128 HUD, optionally
# with a live camera preview between the top/bottom status bars. Modeled on
# the sketches commonly used for ESP32-S3-CAM + ST7735 boards.
# ---------------------------------------------------------------------------

def _display_pixel_hud_includes_and_globals(pins: Dict[str, Any], disp_w: int = 128, disp_h: int = 128) -> str:
    backlight_define = f"#define TFT_BL   {pins['backlight']}" if pins.get("backlight") is not None else ""
    # Precomputed here (not inline in the f-string below) because an
    # f-string expression part cannot contain a backslash on Python < 3.12 -
    # "\n" inside a {...} substitution raises "SyntaxError: f-string
    # expression part cannot include a backslash" on 3.10/3.11.
    backlight_init = (
        "  pinMode(TFT_BL, OUTPUT);\n  digitalWrite(TFT_BL, HIGH);\n"
        if pins.get("backlight") is not None else ""
    )
    return f"""
// ---------------------------------------------------------------------------
// Status display - ST7735 {disp_w}x{disp_h}, pixel-art bordered HUD
// ---------------------------------------------------------------------------
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

#define TFT_MOSI {pins['mosi']}
#define TFT_SCLK {pins['sck']}
#define TFT_CS   {pins['cs']}
#define TFT_DC   {pins['dc']}
#define TFT_RST  {pins['rst']}
{backlight_define}

#define DISP_W        {disp_w}
#define DISP_H        {disp_h}
#define TOP_BAR_H     10
#define BOT_BAR_H     10
#define IMG_Y         TOP_BAR_H
#define IMG_H         (DISP_H - TOP_BAR_H - BOT_BAR_H)

#define PX_BG         0x0000   // black
#define PX_FRAME      0x07FF   // bright cyan
#define PX_ACCENT     0xF81F   // magenta
#define PX_TEXT       0xFFFF   // white
#define PX_GOOD       0x07E0   // green
#define PX_DIM        0x4208   // dark grey-blue

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCLK, TFT_RST);

void drawPixelCornerTick(int x, int y, uint16_t color) {{
  tft.fillRect(x, y, 4, 1, color);
  tft.fillRect(x, y, 1, 4, color);
}}

void drawPixelFrameBorder() {{
  uint16_t c = PX_FRAME;
  for (int x = 0; x < DISP_W; x += 4) {{
    tft.drawPixel(x, IMG_Y - 1, c);
    tft.drawPixel(x, IMG_Y + IMG_H, c);
  }}
  for (int y = IMG_Y - 1; y <= IMG_Y + IMG_H; y += 4) {{
    tft.drawPixel(0, y, c);
    tft.drawPixel(DISP_W - 1, y, c);
  }}
  drawPixelCornerTick(0, IMG_Y - 1, PX_ACCENT);
  drawPixelCornerTick(DISP_W - 4, IMG_Y - 1, PX_ACCENT);
  drawPixelCornerTick(0, IMG_Y + IMG_H - 3, PX_ACCENT);
  drawPixelCornerTick(DISP_W - 4, IMG_Y + IMG_H - 3, PX_ACCENT);
}}

void initDisplay() {{
{backlight_init}  tft.initR(INITR_144GREENTAB);
  tft.setRotation(0);
  tft.fillScreen(PX_BG);
  tft.fillRect(0, 0, DISP_W, TOP_BAR_H, PX_BG);
  tft.drawFastHLine(0, TOP_BAR_H - 1, DISP_W, PX_DIM);
  tft.fillRect(0, DISP_H - BOT_BAR_H, DISP_W, BOT_BAR_H, PX_BG);
  tft.drawFastHLine(0, DISP_H - BOT_BAR_H, DISP_W, PX_DIM);
  drawPixelFrameBorder();
}}

void drawCenteredBootMessage(const String &l1, const String &l2, uint16_t color) {{
  tft.fillScreen(PX_BG);
  tft.setTextColor(color, PX_BG);
  tft.setTextSize(1);
  tft.setCursor((DISP_W - (int16_t)l1.length() * 6) / 2, DISP_H / 2 - 5);
  tft.print(l1);
  if (l2.length() > 0) {{
    tft.setTextColor(PX_DIM, PX_BG);
    tft.setCursor((DISP_W - (int16_t)l2.length() * 6) / 2, DISP_H / 2 + 8);
    tft.print(l2);
  }}
}}

void updateHUDInference(const char* label, float confidence) {{
  tft.fillRect(0, 0, DISP_W, TOP_BAR_H - 1, PX_BG);
  tft.setTextColor(PX_GOOD, PX_BG);
  tft.setCursor(2, 1);
  tft.print(label);

  tft.fillRect(0, DISP_H - BOT_BAR_H + 1, DISP_W, BOT_BAR_H - 1, PX_BG);
  tft.setTextColor(PX_TEXT, PX_BG);
  tft.setCursor(2, DISP_H - 8);
  tft.print("Conf: ");
  tft.print(confidence * 100.0f, 1);
  tft.print("%");
}}

// Text-only prediction update for when no camera is attached (no live
// frame to show inside the frame, so it's left blank between the bars).
void showPredictionOnDisplay(const char* label, float confidence) {{
  tft.fillRect(0, IMG_Y, DISP_W, IMG_H, PX_BG);
  tft.setTextColor(PX_TEXT, PX_BG);
  tft.setTextSize(1);
  tft.setCursor(4, IMG_Y + IMG_H / 2 - 4);
  tft.print(label);
  updateHUDInference(label, confidence);
}}
"""

def _base_ino_header(
    model_name: str, task: str, board: str, input_shape: Tuple[int, ...],
    labels: List[str], arena_bytes: int, display_enabled: bool, display_style: str,
    display_pins: Dict[str, Any], camera_enabled: bool,
    ops: Optional[List[str]] = None, binary_output: bool = False, audio_enabled: bool = False,
) -> str:
    h, w = (input_shape[0], input_shape[1]) if len(input_shape) >= 2 else (96, 96)
    c = input_shape[2] if len(input_shape) >= 3 else 1
    is_od = task == "OBJECT_DETECTION"

    display_lib_note = (
        '//   - "Adafruit GFX Library" and "Adafruit ST7735 and ST7789 Library" (for the status display)'
        if display_enabled else ""
    )
    camera_lib_note = (
        '//   - built-in "esp32-camera" driver (bundled with the esp32 board package)'
        if camera_enabled else ""
    )
    camera_include = '#include "esp_camera.h"\n' if camera_enabled else ""
    audio_include = '#include "mfcc_frontend.h"\n' if audio_enabled else ""
    resolver_include = (
        '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"'
        if ops else '#include "tensorflow/lite/micro/all_ops_resolver.h"'
    )

    if not display_enabled:
        display_block = ""
    elif display_style == "pixel_hud":
        display_block = _display_pixel_hud_includes_and_globals(display_pins)
    else:
        display_block = _display_simple_includes_and_globals(display_pins, camera_enabled, input_shape)

    if is_od:
        print_fn = """
// FOMO-style detector output: a (gridH, gridW, classes+1) grid. Channel 0 is
// background; a cell whose best channel k>0 scores above the threshold holds
// the centre of an object of class k-1. 4-connected cells of the same class
// are merged into one object reported at their probability-weighted centroid
// with the highest cell score - the same decoding as the EdgeCraft backend.
const float kDetectionThreshold = 0.5f;
int lastBestIdx = 0;
float lastBestVal = 0.0f;
int lastDetectionCount = 0;

static int bestClassAt(int gw, int k1, int r, int c, float* score) {
  int best = 0;
  float bestVal = -1.0f;
  for (int k = 0; k < k1; k++) {
    float v = readOutput((r * gw + c) * k1 + k);
    if (v > bestVal) { bestVal = v; best = k; }
  }
  *score = bestVal;
  return best;
}

static int* g_cellClass = nullptr;   // class k>0 of a cell above threshold, 0 = none, -1 = visited
static float* g_cellScore = nullptr;
static int* g_cellStack = nullptr;
static int g_cellCount = 0;

void printPrediction() {
  const int gh = model_output->dims->data[1];
  const int gw = model_output->dims->data[2];
  const int k1 = model_output->dims->data[3];
  const int n = gh * gw;
  if (n > g_cellCount) {
    free(g_cellClass); free(g_cellScore); free(g_cellStack);
    g_cellClass = (int*)malloc(n * sizeof(int));
    g_cellScore = (float*)malloc(n * sizeof(float));
    g_cellStack = (int*)malloc(n * sizeof(int));
    g_cellCount = (g_cellClass && g_cellScore && g_cellStack) ? n : 0;
    if (!g_cellCount) { Serial.println("ERROR: out of memory for detections"); return; }
  }
  for (int i = 0; i < n; i++) {
    float p;
    int k = bestClassAt(gw, k1, i / gw, i % gw, &p);
    g_cellClass[i] = (k > 0 && p >= kDetectionThreshold) ? k : 0;
    g_cellScore[i] = p;
  }
  int count = 0;
  lastBestVal = 0.0f;
  Serial.println("--- Detections ---");
  for (int k = 1; k < k1; k++) {
    for (int seed = 0; seed < n; seed++) {
      if (g_cellClass[seed] != k) continue;
      // Flood-fill one 4-connected blob of class k.
      float sumW = 0.0f, sumR = 0.0f, sumC = 0.0f, maxW = 0.0f;
      int top = 0;
      g_cellStack[top++] = seed;
      g_cellClass[seed] = -1;
      while (top > 0) {
        const int i = g_cellStack[--top];
        const int r = i / gw, c = i % gw;
        const float w = g_cellScore[i];
        sumW += w; sumR += w * r; sumC += w * c;
        if (w > maxW) maxW = w;
        const int nb[4] = {c > 0 ? i - 1 : -1, c < gw - 1 ? i + 1 : -1, r > 0 ? i - gw : -1, r < gh - 1 ? i + gw : -1};
        for (int j = 0; j < 4; j++) {
          if (nb[j] >= 0 && g_cellClass[nb[j]] == k) { g_cellClass[nb[j]] = -1; g_cellStack[top++] = nb[j]; }
        }
      }
      count++;
      const char* name = (k - 1 < kNumLabels) ? kLabels[k - 1] : "object";
      Serial.print("  ");
      Serial.print(name);
      Serial.print(" at x=");
      Serial.print((sumC / sumW + 0.5f) / gw, 3);
      Serial.print(" y=");
      Serial.print((sumR / sumW + 0.5f) / gh, 3);
      Serial.print(" (");
      Serial.print(maxW * 100.0f, 1);
      Serial.println("%)");
      if (maxW > lastBestVal) { lastBestVal = maxW; lastBestIdx = (k - 1 < kNumLabels) ? k - 1 : 0; }
    }
  }
  lastDetectionCount = count;
  Serial.print(count);
  Serial.println(" object(s)");
}
"""
    else:
        print_fn = """
// Prints EVERY class and its confidence to Serial, then the best result.
int lastBestIdx = 0;
float lastBestVal = 0.0f;

void printPrediction() {
  int best_idx = 0;
  float best_val = -1e9f;
  Serial.println("--- Prediction ---");
  for (int i = 0; i < kNumLabels; i++) {
    float v = kIsBinaryOutput ? (i == 1 ? readOutput(0) : 1.0f - readOutput(0)) : readOutput(i);
    Serial.print("  ");
    Serial.print(kLabels[i]);
    Serial.print(": ");
    Serial.print(v * 100.0f, 2);
    Serial.println("%");
    if (v > best_val) { best_val = v; best_idx = i; }
  }
  Serial.print("Best: ");
  Serial.print(kLabels[best_idx]);
  Serial.print("  (");
  Serial.print(best_val * 100.0f, 2);
  Serial.println("%)");
  lastBestIdx = best_idx;
  lastBestVal = best_val;
}
"""

    return f"""\
// {model_name} - generated by EdgeCraft AI
// Task: {task}
// Target board: {board}
// Input shape: {list(input_shape)}
//
// Required Arduino libraries (install via Library Manager):
//   - "Chirale_TensorFlowLite" (provides <Chirale_TensorFlowLite.h>)
//   - ESP32 board package (esp32 by Espressif Systems) >= 2.0.x for ESP32 targets
{camera_lib_note}
{display_lib_note}

#include <Chirale_TensorFlowLite.h>
{resolver_include}
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#if defined(ESP32)
#include "esp_heap_caps.h"
#endif
{camera_include}{audio_include}{display_block}
{_labels_array_cpp(labels)}

namespace {{
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Tensor arena: measured/estimated arena + headroom. If AllocateTensors()
// fails, increase this and re-flash.
constexpr int kTensorArenaSize = {arena_bytes};
uint8_t* tensor_arena = nullptr;
}}  // namespace

const int kInputHeight = {h};
const int kInputWidth  = {w};
const int kInputChannels = {c};
const bool kIsBinaryOutput = {"true" if binary_output else "false"};

// Large buffers go to PSRAM when the board has it (ESP32-S3 / ESP32-CAM);
// a static array that big would not fit in internal DRAM.
void* allocateLarge(size_t bytes) {{
#if defined(ESP32)
  void* p = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p) p = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_8BIT);
  return p;
#else
  return malloc(bytes);
#endif
}}

bool initializeModel() {{
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {{
    MicroPrintf("Model schema version mismatch: %d vs supported %d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }}

  tensor_arena = (uint8_t*)allocateLarge(kTensorArenaSize);
  if (!tensor_arena) {{
    MicroPrintf("Could not allocate a %d byte tensor arena", kTensorArenaSize);
    return false;
  }}

{_op_resolver_block(ops)}

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {{
    MicroPrintf("AllocateTensors() failed - try increasing kTensorArenaSize");
    return false;
  }}
  MicroPrintf("Tensor arena used: %d of %d bytes", (int)interpreter->arena_used_bytes(), kTensorArenaSize);

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  return true;
}}

// Write one float input value (same scale as training: pixels in [0,1],
// MFCCs z-scored), quantizing with rounding + saturation for INT8 models.
void setInputSample(int index, float value) {{
  if (model_input->type == kTfLiteInt8) {{
    int32_t q = (int32_t)lroundf(value / model_input->params.scale) + model_input->params.zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    model_input->data.int8[index] = (int8_t)q;
  }} else {{
    model_input->data.f[index] = value;
  }}
}}

float readOutput(int index) {{
  if (model_output->type == kTfLiteInt8) {{
    return (model_output->data.int8[index] - model_output->params.zero_point) * model_output->params.scale;
  }}
  return model_output->data.f[index];
}}
{print_fn}"""


def _generic_setup_loop(is_audio: bool, display_enabled: bool) -> str:
    display_init = "  initDisplay();\n" if display_enabled else ""
    display_call = (
        "    showPredictionOnDisplay(kLabels[lastBestIdx], lastBestVal);\n"
        if display_enabled else ""
    )

    if is_audio:
        capture = r"""
// ---------------------------------------------------------------------------
// Audio over Serial (no microphone configured)
// ---------------------------------------------------------------------------
// Send the byte 'A' followed by MFCC_N_SAMPLES little-endian int16 samples
// of 16 kHz mono audio (the bundled send_wav.py does exactly this). The
// sketch computes MFCCs on-device with the same front-end used in training.
float* g_audio = nullptr;
float g_features[MFCC_N_MFCC * MFCC_OUT_FRAMES];

bool readSensorFrame() {
  if (Serial.available() < 1 || Serial.read() != 'A') return false;
  for (int n = 0; n < MFCC_N_SAMPLES; n++) {
    uint8_t b[2];
    if (Serial.readBytes(b, 2) != 2) return false;
    g_audio[n] = (int16_t)(b[0] | (b[1] << 8)) / 32768.0f;
  }
  mfcc_compute(g_audio, g_features);
  for (int i = 0; i < MFCC_N_MFCC * MFCC_OUT_FRAMES; i++) setInputSample(i, g_features[i]);
  return true;
}
"""
        ready = "Model ready. Run send_wav.py to stream a clip."
        extra_setup = (
            "  Serial.setTimeout(5000);\n"
            "  g_audio = (float*)allocateLarge(MFCC_N_SAMPLES * sizeof(float));\n"
            "  if (!g_audio) {\n"
            "    Serial.println(\"ERROR: could not allocate the audio buffer\");\n"
            "    while (1) delay(1000);\n"
            "  }\n"
        )
    else:
        capture = r"""
// ---------------------------------------------------------------------------
// Input over Serial (no camera configured)
// ---------------------------------------------------------------------------
// Replace readSensorFrame() with your sensor capture code. As a starting
// point it accepts one frame over Serial: the byte 'I' followed by
// kInputHeight * kInputWidth * kInputChannels raw pixel bytes (0-255,
// row-major, RGB order). EdgeCraft's Device monitor sends exactly this.
bool readSensorFrame() {
  if (Serial.available() < 1 || Serial.read() != 'I') return false;
  const int total = kInputHeight * kInputWidth * kInputChannels;
  for (int i = 0; i < total; i++) {
    uint8_t px;
    if (Serial.readBytes(&px, 1) != 1) return false;
    setInputSample(i, px / 255.0f);
  }
  return true;
}
"""
        ready = "Model ready. Send 'I' + raw pixels (EdgeCraft Device monitor) to run inference."
        extra_setup = "  Serial.setTimeout(5000);\n"

    return capture + f"""
void setup() {{
  Serial.begin(115200);
  delay(1000);
  Serial.println("EdgeCraft AI - model init...");
{display_init}{extra_setup}  if (!initializeModel()) {{
    Serial.println("ERROR: model init failed");
    while (1) delay(1000);
  }}
  Serial.println("{ready}");
}}

void loop() {{
  if (readSensorFrame()) {{
    unsigned long t0 = micros();
    if (interpreter->Invoke() != kTfLiteOk) {{
      Serial.println("ERROR: inference failed");
      return;
    }}
    unsigned long dt = micros() - t0;
    Serial.print("Inference: ");
    Serial.print(dt);
    Serial.println(" us");
    printPrediction();
{display_call}  }}
}}
"""


def _i2s_mic_setup_loop(mic_pins: Dict[str, int], display_enabled: bool) -> str:
    display_init = "  initDisplay();\n" if display_enabled else ""
    display_call = "  showPredictionOnDisplay(kLabels[lastBestIdx], lastBestVal);\n" if display_enabled else ""
    return f"""
// ---------------------------------------------------------------------------
// I2S microphone (e.g. INMP441 / SPH0645) -> on-device MFCC -> model
// ---------------------------------------------------------------------------
#include <driver/i2s.h>
#define I2S_MIC_SCK {mic_pins['sck']}
#define I2S_MIC_WS  {mic_pins['ws']}
#define I2S_MIC_SD  {mic_pins['sd']}

float* g_audio = nullptr;
float g_features[MFCC_N_MFCC * MFCC_OUT_FRAMES];

bool initMic() {{
  i2s_config_t cfg = {{}};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = MFCC_SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  i2s_pin_config_t pins = {{}};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = I2S_MIC_SCK;
  pins.ws_io_num = I2S_MIC_WS;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = I2S_MIC_SD;
  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL) != ESP_OK) return false;
  return i2s_set_pin(I2S_NUM_0, &pins) == ESP_OK;
}}

// Records one clip (MFCC_N_SAMPLES at 16 kHz) into g_audio.
void recordClip() {{
  int32_t buf[256];
  int n = 0;
  while (n < MFCC_N_SAMPLES) {{
    size_t bytes = 0;
    i2s_read(I2S_NUM_0, buf, sizeof(buf), &bytes, portMAX_DELAY);
    for (size_t i = 0; i < bytes / 4 && n < MFCC_N_SAMPLES; i++) {{
      g_audio[n++] = (buf[i] >> 8) / 8388608.0f;  // 24-bit sample in a 32-bit slot
    }}
  }}
}}

void setup() {{
  Serial.begin(115200);
  delay(1000);
  Serial.println("EdgeCraft AI - microphone + model init...");
{display_init}  g_audio = (float*)allocateLarge(MFCC_N_SAMPLES * sizeof(float));
  if (!g_audio || !initMic()) {{
    Serial.println("ERROR: microphone init failed - check I2S wiring");
    while (1) delay(1000);
  }}
  if (!initializeModel()) {{
    Serial.println("ERROR: model init failed");
    while (1) delay(1000);
  }}
  Serial.println("Listening...");
}}

void loop() {{
  recordClip();
  unsigned long t0 = micros();
  mfcc_compute(g_audio, g_features);
  for (int i = 0; i < MFCC_N_MFCC * MFCC_OUT_FRAMES; i++) setInputSample(i, g_features[i]);
  unsigned long t1 = micros();
  if (interpreter->Invoke() != kTfLiteOk) {{
    Serial.println("ERROR: inference failed");
    return;
  }}
  Serial.printf("MFCC: %lu us, inference: %lu us\\n", t1 - t0, micros() - t1);
  printPrediction();
{display_call}}}
"""


SEND_WAV_SCRIPT = r"""#!/usr/bin/env python3
# Stream a WAV clip to an EdgeCraft sketch over Serial: python send_wav.py COM3 clip.wav
# Needs `pip install pyserial numpy`. The clip is converted to 16 kHz mono int16,
# padded/trimmed to the model's clip length, then sent as 'A' + raw samples.
import sys, time, wave
import numpy as np
import serial

N_SAMPLES = __N_SAMPLES__
port, path = sys.argv[1], sys.argv[2]
with wave.open(path) as w:
    sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(w.getnframes())
data = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width]).astype(np.float32)
if width == 1:
    data = (data - 128) / 128
else:
    data /= float(2 ** (8 * width - 1))
data = data.reshape(-1, ch).mean(axis=1)
if sr != 16000:
    data = np.interp(np.arange(0, len(data), sr / 16000.0), np.arange(len(data)), data)
data = np.pad(data[:N_SAMPLES], (0, max(0, N_SAMPLES - len(data))))
pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
with serial.Serial(port, 115200, timeout=1) as s:
    time.sleep(2)  # board resets when the port opens
    s.write(b"A" + pcm)
    end = time.time() + 5
    while time.time() < end:
        line = s.readline().decode(errors="replace").rstrip()
        if line:
            print(line)
"""


def _camera_pins_defines(pins: Dict[str, int]) -> str:
    return (
        f"#define PWDN_GPIO_NUM     {pins['pwdn']}\n"
        f"#define RESET_GPIO_NUM    {pins['reset']}\n"
        f"#define XCLK_GPIO_NUM     {pins['xclk']}\n"
        f"#define SIOD_GPIO_NUM     {pins['siod']}\n"
        f"#define SIOC_GPIO_NUM     {pins['sioc']}\n"
        f"#define Y9_GPIO_NUM       {pins['y9']}\n"
        f"#define Y8_GPIO_NUM       {pins['y8']}\n"
        f"#define Y7_GPIO_NUM       {pins['y7']}\n"
        f"#define Y6_GPIO_NUM       {pins['y6']}\n"
        f"#define Y5_GPIO_NUM       {pins['y5']}\n"
        f"#define Y4_GPIO_NUM       {pins['y4']}\n"
        f"#define Y3_GPIO_NUM       {pins['y3']}\n"
        f"#define Y2_GPIO_NUM       {pins['y2']}\n"
        f"#define VSYNC_GPIO_NUM    {pins['vsync']}\n"
        f"#define HREF_GPIO_NUM     {pins['href']}\n"
        f"#define PCLK_GPIO_NUM     {pins['pclk']}"
    )

def _camera_init_block(pixfmt: str) -> str:
    return f"""bool initCamera() {{
  // Zero-initialised: camera_config_t has fields not set below (sccb_i2c_port,
  // jpeg_buffer_size, ...) that would otherwise hold stack garbage.
  camera_config_t config = {{}};
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
  config.frame_size = FRAMESIZE_QQVGA;   // 160x120 - cropped/downsampled in software to the model's exact input
  config.pixel_format = {pixfmt};

  if (psramFound()) {{
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
  }} else {{
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }}

  if (esp_camera_init(&config) != ESP_OK) {{
    Serial.println("ERROR: camera init failed");
    return false;
  }}

  // Only the OV3660 is mounted upside-down on these boards (same as
  // Espressif's CameraWebServer example); flipping an OV2640 would feed the
  // model upside-down frames.
  sensor_t *s = esp_camera_sensor_get();
  if (s && s->id.PID == OV3660_PID) {{
    s->set_vflip(s, 1);
  }}
  return true;
}}"""

def _camera_setup_loop_simple(
    input_shape: Tuple[int, ...], camera_pins: Dict[str, int],
    display_enabled: bool, camera_module_type: str,
) -> str:
    """Original "simple" body: capture -> downsample straight into the
    model's input tensor -> inference -> Serial + optional text/preview HUD."""
    h, w = input_shape[0], input_shape[1]
    channels = input_shape[2] if len(input_shape) >= 3 else 1
    is_rgb = channels == 3
    pixfmt = "PIXFORMAT_RGB565" if is_rgb else "PIXFORMAT_GRAYSCALE"
    display_init = "  initDisplay();\n" if display_enabled else ""

    module_note = (
        "Integrated camera module (e.g. ESP32-CAM/ESP32-S3-CAM's onboard sensor)."
        if camera_module_type == "integrated"
        else "External camera module wired to this board's GPIO pins."
    )

    display_buffer_decl = f"uint16_t displayFrameBuffer[{w} * {h}];\n" if display_enabled else ""
    display_pixel_write = (
        "      displayFrameBuffer[idx] = tft.color565(gray, gray, gray);\n"
        if display_enabled and not is_rgb
        else ("      displayFrameBuffer[idx] = rgb565;\n" if display_enabled and is_rgb else "")
    )
    display_call = (
        f"  showCameraFrameAndPrediction(displayFrameBuffer, {w}, {h}, kLabels[lastBestIdx], lastBestVal);\n"
        if display_enabled else ""
    )

    if is_rgb:
        pixel_extract = """      // RGB565, 2 bytes per source pixel. ESP32 camera frames are typically
      // byte-swapped relative to what Adafruit_GFX expects for direct
      // drawRGBBitmap() use - if colours look wrong on your display, try
      // removing the byte swap below (or vice versa).
      int src_idx = (sy * src_w + sx) * 2;
      uint16_t raw565 = (fb->buf[src_idx] << 8) | fb->buf[src_idx + 1];
      uint8_t r5 = (raw565 >> 11) & 0x1F;
      uint8_t g6 = (raw565 >> 5) & 0x3F;
      uint8_t b5 = raw565 & 0x1F;
      float rNorm = (r5 * 255 / 31) / 255.0f;
      float gNorm = (g6 * 255 / 63) / 255.0f;
      float bNorm = (b5 * 255 / 31) / 255.0f;
      setInputSample(idx * 3 + 0, rNorm);
      setInputSample(idx * 3 + 1, gNorm);
      setInputSample(idx * 3 + 2, bNorm);
      uint16_t rgb565 = raw565;
"""
    else:
        pixel_extract = """      uint8_t gray = fb->buf[sy * src_w + sx];  // grayscale, 1 byte per pixel
      setInputSample(idx, gray / 255.0f);
"""

    return f"""
// ---------------------------------------------------------------------------
// Camera capture - {module_note}
// Pins configured in the Deployment tab (defaults match the selected
// camera module preset; edit them there if your wiring differs).
// ---------------------------------------------------------------------------
{_camera_pins_defines(camera_pins)}

{display_buffer_decl}
{_camera_init_block(pixfmt)}

// Nearest-neighbour downsample from the captured frame to the model's
// {w}x{h}x{channels} input, normalising pixel values to [0,1]. When a
// display is attached, this ALSO fills displayFrameBuffer with the same
// downsampled frame in RGB565 so it can be shown as a live preview -
// one pass powers both the model input and the on-screen image.
void frameToModelInput(camera_fb_t* fb) {{
  int src_w = fb->width;
  int src_h = fb->height;
  int idx = 0;
  for (int y = 0; y < {h}; y++) {{
    int sy = y * src_h / {h};
    for (int x = 0; x < {w}; x++) {{
      int sx = x * src_w / {w};
{pixel_extract}{display_pixel_write}      idx++;
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
  Serial.println("Camera + model ready. Streaming live inference...");
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

def _camera_setup_loop_pixel_hud(
    input_shape: Tuple[int, ...], camera_pins: Dict[str, int],
    camera_module_type: str, disp_w: int = 128, disp_h: int = 128,
) -> str:
    """Pixel-HUD body: JPEG capture -> jpg2rgb565 -> crop/downsample both
    into the model's input tensor AND onto the framed TFT display, with a
    top/bottom status bar HUD. Mirrors the layout used on common
    ESP32-S3-CAM + ST7735 project sketches (larger source frame captured as
    JPEG, decoded to RGB565, cropped to the display, separately downsampled
    to the model's resolution)."""
    h, w = input_shape[0], input_shape[1]
    channels = input_shape[2] if len(input_shape) >= 3 else 1
    is_rgb = channels == 3
    src_w, src_h = 160, 120  # QQVGA - decoded once, reused for both display + model input

    if is_rgb:
        model_pixel_extract = """      uint16_t pixel = rgbBuf[sy * src_w + sx];
      uint8_t r5 = (pixel >> 11) & 0x1F;
      uint8_t g6 = (pixel >> 5) & 0x3F;
      uint8_t b5 = pixel & 0x1F;
      float rNorm = (r5 * 255 / 31) / 255.0f;
      float gNorm = (g6 * 255 / 63) / 255.0f;
      float bNorm = (b5 * 255 / 31) / 255.0f;
      setInputSample(idx * 3 + 0, rNorm);
      setInputSample(idx * 3 + 1, gNorm);
      setInputSample(idx * 3 + 2, bNorm);
"""
    else:
        model_pixel_extract = """      uint16_t pixel = rgbBuf[sy * src_w + sx];
      uint8_t r5 = (pixel >> 11) & 0x1F;
      uint8_t g6 = (pixel >> 5) & 0x3F;
      uint8_t b5 = pixel & 0x1F;
      // Convert to grayscale via standard luma weights since the model
      // expects a single channel even though the sensor only outputs RGB.
      float gray = (0.299f * (r5 * 255 / 31) + 0.587f * (g6 * 255 / 63) + 0.114f * (b5 * 255 / 31)) / 255.0f;
      setInputSample(idx, gray);
"""

    return f"""
// ---------------------------------------------------------------------------
// Camera capture ({"integrated module" if camera_module_type == "integrated" else "external module wired to GPIO"})
// Pins configured in the Deployment tab (defaults match the selected
// camera module preset; edit them there if your wiring differs).
// ---------------------------------------------------------------------------
#include "img_converters.h"   // required for jpg2rgb565 decoding
{_camera_pins_defines(camera_pins)}

bool initCamera() {{
  camera_config_t cfg = {{}};  // zero-init the fields not set below
  cfg.ledc_channel  = LEDC_CHANNEL_0;
  cfg.ledc_timer    = LEDC_TIMER_0;
  cfg.pin_d0        = Y2_GPIO_NUM;
  cfg.pin_d1        = Y3_GPIO_NUM;
  cfg.pin_d2        = Y4_GPIO_NUM;
  cfg.pin_d3        = Y5_GPIO_NUM;
  cfg.pin_d4        = Y6_GPIO_NUM;
  cfg.pin_d5        = Y7_GPIO_NUM;
  cfg.pin_d6        = Y8_GPIO_NUM;
  cfg.pin_d7        = Y9_GPIO_NUM;
  cfg.pin_xclk      = XCLK_GPIO_NUM;
  cfg.pin_sscb_scl  = SIOC_GPIO_NUM;
  cfg.pin_sscb_sda  = SIOD_GPIO_NUM;
  cfg.pin_vsync     = VSYNC_GPIO_NUM;
  cfg.pin_href      = HREF_GPIO_NUM;
  cfg.pin_pclk      = PCLK_GPIO_NUM;
  cfg.pin_pwdn      = PWDN_GPIO_NUM;
  cfg.pin_reset     = RESET_GPIO_NUM;
  cfg.xclk_freq_hz  = 20000000;
  cfg.pixel_format  = PIXFORMAT_JPEG;

  if (psramFound()) {{
    cfg.frame_size   = FRAMESIZE_QQVGA;   // {src_w}x{src_h}
    cfg.jpeg_quality = 14;
    cfg.fb_count     = 2;
    cfg.fb_location  = CAMERA_FB_IN_PSRAM;
    cfg.grab_mode    = CAMERA_GRAB_LATEST;
  }} else {{
    cfg.frame_size   = FRAMESIZE_QQVGA;
    cfg.jpeg_quality = 16;
    cfg.fb_count     = 1;
    cfg.fb_location  = CAMERA_FB_IN_DRAM;
    cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
  }}

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) return false;

  sensor_t *s = esp_camera_sensor_get();
  if (s) {{
    if (s->id.PID == OV3660_PID) s->set_vflip(s, 1);  // OV3660 is mounted upside-down
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
  }}
  return true;
}}

// Downsamples the decoded RGB565 frame into the model's {w}x{h}x{channels}
// input tensor.
void frameToModelInput(const uint16_t* rgbBuf) {{
  int src_w = {src_w};
  int src_h = {src_h};
  int idx = 0;
  for (int y = 0; y < {h}; y++) {{
    int sy = y * src_h / {h};
    for (int x = 0; x < {w}; x++) {{
      int sx = x * src_w / {w};
{model_pixel_extract}      idx++;
    }}
  }}
}}

void pipelineCaptureAndProcess() {{
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  static uint16_t rgbBuf[{src_w} * {src_h}] __attribute__((aligned(32)));
  if (!jpg2rgb565(fb->buf, fb->len, (uint8_t *)rgbBuf, JPG_SCALE_NONE)) {{
    esp_camera_fb_return(fb);
    return;
  }}
  esp_camera_fb_return(fb);

  // 1. Crop and push the display-resolution window to the TFT
  const int offsetX = ({src_w} - DISP_W) / 2;
  const int offsetY = ({src_h} - IMG_H) / 2;
  const int drawH   = IMG_H;

  tft.startWrite();
  tft.setAddrWindow(0, IMG_Y, DISP_W, drawH);
  for (int y = 0; y < drawH; y++) {{
    int srcRow = y + offsetY;
    tft.writePixels(&rgbBuf[srcRow * {src_w} + offsetX], DISP_W, false);
  }}
  tft.endWrite();

  // 2. Downsample the same frame into the model's input tensor
  frameToModelInput(rgbBuf);

  // 3. Run inference and update the HUD
  if (interpreter->Invoke() == kTfLiteOk) {{
    printPrediction();
    updateHUDInference(kLabels[lastBestIdx], lastBestVal);
  }}
}}

void setup() {{
  Serial.begin(115200);
  delay(300);
  Serial.println("EdgeCraft AI - camera + display + model init...");

  initDisplay();
  drawCenteredBootMessage("BOOTING", "Initializing Camera...", PX_FRAME);
  if (!initCamera()) {{
    drawCenteredBootMessage("CAM FAIL", "Check hardware setup", 0xF800);
    while (1) delay(1000);
  }}

  drawCenteredBootMessage("BOOTING", "Loading AI Engine...", PX_FRAME);
  if (!initializeModel()) {{
    drawCenteredBootMessage("ML FAIL", "Check model_data.h integrity", 0xF800);
    while (1) delay(1000);
  }}

  Serial.println("Ready. Streaming live inference...");
}}

void loop() {{
  pipelineCaptureAndProcess();
}}
"""

def generate_arduino_sketch(
    model_name: str, task: str, board: str, input_shape: Tuple[int, ...],
    labels: List[str], arena_bytes: int,
    camera_pins: Optional[Dict[str, int]] = None,
    display_config: Optional[Dict[str, Any]] = None,
    camera_config: Optional[Dict[str, Any]] = None,
    mic_config: Optional[Dict[str, Any]] = None,
    ops: Optional[List[str]] = None,
    binary_output: bool = False,
) -> str:
    is_audio = task in _AUDIO_TASKS
    camera_enabled = _resolve_camera_enabled(board, camera_config) and not is_audio
    resolved_camera_pins, camera_module_type, _camera_preset_id = _resolve_camera_pins(camera_config, camera_pins)
    display_enabled, display_pins, display_style, _display_preset_id = _resolve_display(display_config)
    mic_enabled, mic_pins, _mic_preset = _resolve_mic(board, mic_config) if is_audio else (False, {}, "NONE")

    header = _base_ino_header(
        model_name, task, board, input_shape, labels, arena_bytes,
        display_enabled, display_style, display_pins, camera_enabled,
        ops=ops, binary_output=binary_output, audio_enabled=is_audio,
    )

    if is_audio and mic_enabled:
        body = _i2s_mic_setup_loop(mic_pins, display_enabled)
    elif camera_enabled and display_enabled and display_style == "pixel_hud":
        body = _camera_setup_loop_pixel_hud(input_shape, resolved_camera_pins, camera_module_type)
    elif camera_enabled:
        body = _camera_setup_loop_simple(input_shape, resolved_camera_pins, display_enabled, camera_module_type)
    elif display_enabled and display_style == "pixel_hud":
        # No camera: same bordered HUD frame, text-only prediction.
        body = _generic_setup_loop(is_audio=task in _AUDIO_TASKS, display_enabled=True)
    else:
        body = _generic_setup_loop(is_audio=task in _AUDIO_TASKS, display_enabled=display_enabled)

    return header + body

def _generate_readme(
    board: str, task: str, method: str, arena_bytes: int,
    input_shape: Tuple[int, ...], labels: List[str],
    camera_pins: Dict[str, int], display_pins: Dict[str, Any],
    camera_enabled: bool, camera_module_type: str, camera_preset_id: str,
    display_enabled: bool, display_style: str, display_preset_id: str,
    audio: str = "", tflm: Optional[Dict[str, Any]] = None,
) -> str:
    tflm = tflm or {}
    if tflm.get("supported") and tflm.get("arena_bytes"):
        arena_note = (f"Verified with the TensorFlow Lite Micro interpreter on the host: the model needs "
                      f"{tflm['arena_bytes'] // 1024} KB of arena; the sketch reserves {arena_bytes // 1024} KB.")
    else:
        arena_note = (f"Estimated from a liveness analysis of the graph ({arena_bytes // 1024} KB reserved). "
                      "If `AllocateTensors()` fails, increase `kTensorArenaSize` and re-flash.")
    if camera_enabled:
        camera_label = CAMERA_MODULE_PRESETS.get(camera_preset_id, {}).get("label", camera_preset_id)
        camera_note = f"""
This sketch is configured for **{camera_label}**. It captures frames
continuously and runs live inference on each one - see the
`#define ..._GPIO_NUM` block at the top of `sketch.ino` for the pin
assignments (these match the camera module preset you chose in the
Deployment tab; double-check them against your actual wiring if you used a
custom module).
"""
    elif audio:
        camera_note = audio
    else:
        camera_note = (
            "\nThis board has no camera configured. The sketch reads test frames\n"
            "over Serial ('I' + raw pixel bytes - the app's Device monitor sends them) so you can validate\n"
            "inference before wiring up your real sensor/microphone capture code\n"
            "in `readSensorFrame()`.\n"
        )

    display_section = ""
    if display_enabled:
        display_label = DISPLAY_MODULE_PRESETS.get(display_preset_id, {}).get("label", display_preset_id)
        if display_style == "pixel_hud":
            live_preview_note = (
                "\nSince a camera is also attached, the frame between the top/bottom bars\n"
                "shows a **live preview** of the captured camera feed, with the predicted\n"
                "class and confidence in the status bars, refreshed every inference\n"
                "cycle.\n"
                if camera_enabled else
                "\nNo camera is attached, so the frame between the bars stays blank -\n"
                "only the predicted class and confidence in the status bars update.\n"
            )
        else:
            live_preview_note = (
                "\nSince a camera is also attached, the display shows a **live preview**\n"
                "of the downsampled camera frame plus the current prediction and\n"
                "confidence, refreshed every inference cycle - a basic live camera +\n"
                "inference UI.\n"
                if camera_enabled else
                "\nShows the current prediction and confidence as text (no camera attached,\n"
                "so there's no live video feed to preview).\n"
            )
        display_section = f"""
## Status display
**{display_label}**, wired with these pins (as configured in the Deployment tab):
- CS:   {display_pins.get('cs')}
- DC:   {display_pins.get('dc')}
- RST:  {display_pins.get('rst')}
- SCK:  {display_pins.get('sck')}
- MOSI: {display_pins.get('mosi')}
{f"- Backlight: {display_pins['backlight']}" if display_pins.get('backlight') is not None else "- Backlight: tied directly to 3.3V (no GPIO control configured)"}
{live_preview_note}
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
- Tensor arena: {arena_bytes // 1024} KB, allocated in PSRAM when available.
  {arena_note}

## Required Arduino libraries
1. Board package: **esp32** by Espressif Systems (Boards Manager) - version 2.0.x or newer
2. Library: **Chirale_TensorFlowLite** - search that exact name in Library Manager
   (Sketch > Include Library > Manage Libraries...). It installs a header called
   `Chirale_TensorFlowLite.h`, which is what `sketch.ino` includes.
{"3. Library: built-in **esp32-camera** driver (bundled with the esp32 board package)" if camera_enabled else ""}
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
"Model ready" (or "Camera + model ready" if a camera is attached), followed
by a full `--- Prediction ---` block every inference cycle listing **every**
class and its confidence percentage, then a `Best: <label> (<confidence>%)`
summary line.
{"The status display mirrors this with the live camera preview and the current best prediction." if display_enabled and camera_enabled else ("The status display mirrors the current best prediction as text." if display_enabled else "")}

## Notes
- The sketch registers only the ops the model uses (`MicroMutableOpResolver`),
  which keeps flash usage down. It prints the arena actually used at boot.
- `Inference: N us` lines report real on-device latency.
"""

def _audio_readme(ctx: Dict[str, Any], mic_enabled: bool, mic_pins: Dict[str, int]) -> str:
    n_mfcc, frames = int(ctx["input_shape"][0]), int(ctx["input_shape"][1])
    p = preprocessing.audio_params(ctx["task"], n_mfcc)
    common = (
        f"\nAudio is turned into features on-device by `mfcc_frontend.h`, a port of the exact "
        f"training front-end ({n_mfcc} MFCCs x {frames} frames from a {p['duration']} s, 16 kHz clip).\n"
    )
    if mic_enabled:
        return common + (
            f"\nAn I2S microphone (INMP441-style) is expected on SCK={mic_pins['sck']}, "
            f"WS={mic_pins['ws']}, SD={mic_pins['sd']} (L/R pin to GND). The sketch records "
            "a clip, computes MFCCs and classifies it in a loop.\n"
        )
    return common + (
        "\nNo microphone is configured: stream a WAV file from your PC with\n"
        "`python send_wav.py <serial-port> clip.wav` (needs `pip install pyserial numpy`).\n"
    )


# ---------------------------------------------------------------------------
# Full package (zip)
# ---------------------------------------------------------------------------

def generate_export_package(
    optimization_id: str, board: str,
    camera_pins: Optional[Dict[str, int]] = None,
    display_config: Optional[Dict[str, Any]] = None,
    camera_config: Optional[Dict[str, Any]] = None,
    mic_config: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Build a zip containing model_data.h, sketch.ino, and README.md,
    wired for the user's actual hardware configuration (an optional camera -
    picked from a named preset or hand-wired - and an optional attached
    status display, also picked from a named preset)."""
    ctx = _get_export_context(optimization_id)

    c_header = CArrayGenerator.binary_to_c_array(ctx["tflite_bytes"], model_name="model")
    arena_bytes = _estimate_arena_bytes(optimization_id, board)
    is_audio = ctx["task"] in _AUDIO_TASKS

    camera_enabled = _resolve_camera_enabled(board, camera_config) and not is_audio
    resolved_camera_pins, camera_module_type, camera_preset_id = _resolve_camera_pins(camera_config, camera_pins)
    display_enabled, display_pins, display_style, display_preset_id = _resolve_display(display_config)

    sketch = generate_arduino_sketch(
        model_name="model",
        task=ctx["task"],
        board=board,
        input_shape=ctx["input_shape"],
        labels=ctx["labels"],
        arena_bytes=arena_bytes,
        camera_pins=camera_pins,
        display_config=display_config,
        camera_config=camera_config,
        mic_config=mic_config,
        ops=ctx["ops"],
        binary_output=ctx["binary_output"],
    )
    mic_enabled, mic_pins, _ = _resolve_mic(board, mic_config) if is_audio else (False, {}, "NONE")

    readme = _generate_readme(
        board=board,
        task=ctx["task"],
        method=ctx["method"],
        arena_bytes=arena_bytes,
        input_shape=ctx["input_shape"],
        labels=ctx["labels"],
        camera_pins=resolved_camera_pins,
        display_pins=display_pins,
        camera_enabled=camera_enabled,
        camera_module_type=camera_module_type,
        camera_preset_id=camera_preset_id,
        display_enabled=display_enabled,
        display_style=display_style,
        display_preset_id=display_preset_id,
        audio=_audio_readme(ctx, mic_enabled, mic_pins) if is_audio else "",
        tflm=ctx["tflm"],
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model_data.h", c_header)
        zf.writestr("sketch.ino", sketch)
        zf.writestr("README.md", readme)
        if is_audio:
            n_mfcc, frames = int(ctx["input_shape"][0]), int(ctx["input_shape"][1])
            zf.writestr("mfcc_frontend.h", generate_mfcc_header(ctx["task"], n_mfcc, frames))
            n_samples = preprocessing.audio_params(ctx["task"], n_mfcc)["n_samples"]
            zf.writestr("send_wav.py", SEND_WAV_SCRIPT.replace("__N_SAMPLES__", str(int(n_samples))))
    buf.seek(0)
    return buf.read()

def preview_sketch(
    optimization_id: str, board: str,
    camera_pins: Optional[Dict[str, int]] = None,
    display_config: Optional[Dict[str, Any]] = None,
    camera_config: Optional[Dict[str, Any]] = None,
    mic_config: Optional[Dict[str, Any]] = None,
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
        camera_config=camera_config,
        mic_config=mic_config,
        ops=ctx["ops"],
        binary_output=ctx["binary_output"],
    )