"""Generated Arduino sketch: TFLM API names, board portability and the
on-device FOMO decoder (compiled with a host C++ compiler against tiny stubs)."""

import shutil
import subprocess

import numpy as np
import pytest

from app.services import exporter
from app.services.detection import decode_predictions
from app.utils.c_array_generator import CArrayGenerator


def test_resolver_method_names_for_irregular_ops():
    # The CamelCase rule gets these wrong (AddBatchMatmul / AddCumsum / AddPadv2
    # don't exist in MicroMutableOpResolver), which broke the sketch build.
    assert exporter._resolver_method("BATCH_MATMUL") == "AddBatchMatMul"
    assert exporter._resolver_method("CUMSUM") == "AddCumSum"
    assert exporter._resolver_method("PADV2") == "AddPadV2"
    assert exporter._resolver_method("REVERSE_V2") == "AddReverseV2"


def test_op_resolver_block_deduplicates_ops():
    block = exporter._op_resolver_block(["CONV_2D", "SOFTMAX", "CONV_2D", "DELEGATE"])
    assert "MicroMutableOpResolver<2>" in block
    assert block.count("AddConv2D") == 1


def test_model_array_is_16_byte_aligned():
    header = CArrayGenerator.binary_to_c_array(bytes(range(40)), model_name="model")
    assert "aligned(16)" in header and "g_model_len = 40;" in header


def _sketch(task, board, input_shape, **kw):
    return exporter.generate_arduino_sketch(
        "model", task, board, input_shape, ["a", "b"], 32768, ops=["CONV_2D", "SOFTMAX"], **kw)


@pytest.mark.parametrize("task,shape", [("IMAGE_CLASSIFICATION", (32, 32, 3)), ("OBJECT_DETECTION", (32, 32, 3)),
                                        ("KEYWORD_SPOTTING", (40, 49, 1))])
def test_non_esp32_sketch_avoids_serial_printf(task, shape):
    # ArduinoCore-API's Print (mbed boards like the Nano 33 BLE) has no printf().
    assert "Serial.printf" not in _sketch(task, "ARDUINO_NANO_33_BLE", shape)


@pytest.mark.parametrize("display", [{"enabled": True},
                                     {"enabled": True, "module_preset": "ST7735_PIXEL_HUD"}])
def test_camera_config_is_valid_and_initialised(display):
    sketch = _sketch("IMAGE_CLASSIFICATION", "ESP32_CAM", (32, 32, 3), display_config=display)
    # camera_config_t has no pin_sioc/pin_siod members (compile error), and
    # unset fields (fb_location, jpeg_buffer_size, ...) must not be garbage.
    assert "pin_sioc" not in sketch and "pin_siod" not in sketch
    assert "camera_config_t config = {};" in sketch or "camera_config_t cfg = {};" in sketch
    assert "CAMERA_FB_IN_PSRAM" in sketch and "CAMERA_FB_IN_DRAM" in sketch
    # Only the OV3660 needs a vertical flip; the ESP32-CAM's OV2640 must not be flipped.
    assert "OV3660_PID" in sketch


STUBS = {
    "Chirale_TensorFlowLite.h": r"""
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
struct FakeSerial {
  void print(const char* s) { printf("%s", s); }
  void print(int v) { printf("%d", v); }
  void print(double v, int d) { printf("%.*f", d, v); }
  void println(const char* s) { printf("%s\n", s); }
};
extern FakeSerial Serial;
""",
    "tensorflow/lite/micro/micro_mutable_op_resolver.h": r"""
#pragma once
namespace tflite { template <unsigned int N> struct MicroMutableOpResolver { int AddConv2D() { return 0; } int AddSoftmax() { return 0; } }; }
""",
    "tensorflow/lite/micro/micro_interpreter.h": r"""
#pragma once
#include <stddef.h>
#include <stdint.h>
enum TfLiteType { kTfLiteFloat32 = 1, kTfLiteInt8 = 9 };
enum TfLiteStatus { kTfLiteOk, kTfLiteError };
struct TfLiteIntArray { int size; int data[4]; };
struct TfLiteQuantizationParams { float scale; int32_t zero_point; };
union TfLitePtrUnion { float* f; int8_t* int8; };
struct TfLiteTensor { TfLiteType type; TfLitePtrUnion data; TfLiteIntArray* dims; TfLiteQuantizationParams params; };
namespace tflite {
struct Model { int version() const { return 3; } };
inline const Model* GetModel(const void*) { static Model m; return &m; }
struct MicroInterpreter {
  template <typename R> MicroInterpreter(const Model*, R&, uint8_t*, size_t) {}
  TfLiteStatus AllocateTensors() { return kTfLiteOk; }
  size_t arena_used_bytes() { return 0; }
  TfLiteTensor* input(size_t);
  TfLiteTensor* output(size_t);
};
}
""",
    "tensorflow/lite/micro/micro_log.h": "#pragma once\n#define MicroPrintf(...) ((void)0)\n",
    "tensorflow/lite/schema/schema_generated.h": "#pragma once\n#define TFLITE_SCHEMA_VERSION 3\n",
}

HARNESS = r"""
FakeSerial Serial;
static TfLiteIntArray g_dims = {4, {1, GH, GW, K1}};
static float g_out[GH * GW * K1];
static TfLiteTensor g_tensor = {kTfLiteFloat32, {g_out}, &g_dims, {0.0f, 0}};
TfLiteTensor* tflite::MicroInterpreter::output(size_t) { return &g_tensor; }
TfLiteTensor* tflite::MicroInterpreter::input(size_t) { return &g_tensor; }
int main() {
  model_output = &g_tensor;
  for (int i = 0; i < GH * GW * K1; i++) if (scanf("%f", &g_out[i]) != 1) return 1;
  printPrediction();
  return 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
def test_fomo_decoder_matches_backend(tmp_path):
    gh, gw, k1 = 6, 6, 3
    sketch = exporter.generate_arduino_sketch(
        "model", "OBJECT_DETECTION", "ARDUINO_NANO_33_BLE", (48, 48, 3), ["chip", "led"], 32768,
        ops=["CONV_2D", "SOFTMAX"])
    # Drop setup()/loop(): only the decoder is exercised.
    sketch = sketch[:sketch.index("// Input over Serial")]
    for name, text in STUBS.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (tmp_path / "model_data.h").write_text(CArrayGenerator.binary_to_c_array(b"\x00" * 16, "model"))
    (tmp_path / "main.cpp").write_text(sketch + HARNESS)
    exe = tmp_path / "fomo"
    subprocess.run(["g++", "-std=gnu++17", f"-DGH={gh}", f"-DGW={gw}", f"-DK1={k1}", "-I", str(tmp_path),
                    "-o", str(exe), str(tmp_path / "main.cpp")], check=True)

    def background():
        g = np.zeros((gh, gw, k1), dtype=np.float32)
        g[..., 0] = 0.9
        g[..., 1:] = 0.1 / (k1 - 1)
        return g

    def obj(g, r, c, k, p):
        g[r, c] = (1 - p) / (k1 - 1)
        g[r, c, k] = p

    grids = []
    g = background()  # an L-shaped blob that a left/up-neighbour check reports twice
    for (r, c), p in zip([(0, 2), (1, 1), (1, 2)], [0.7, 0.9, 0.8]):
        obj(g, r, c, 1, p)
    obj(g, 4, 4, 2, 0.95)  # a second class
    obj(g, 3, 0, 1, 0.6)   # diagonal to nothing: its own object
    obj(g, 4, 1, 1, 0.55)  # diagonal neighbour of (3, 0): separate under 4-connectivity
    grids.append(g)
    rng = np.random.default_rng(3)
    for _ in range(5):
        logits = rng.normal(size=(gh, gw, k1)) * 2
        logits[..., 0] += 1.0
        e = np.exp(logits)
        grids.append((e / e.sum(-1, keepdims=True)).astype(np.float32))

    for g in grids:
        run = subprocess.run([str(exe)], input=" ".join(f"{v:.8f}" for v in g.reshape(-1)),
                             capture_output=True, text=True, check=True)
        device = []
        for line in run.stdout.splitlines():
            if " at x=" in line:
                name, rest = line.strip().split(" at x=")
                x, rest = rest.split(" y=")
                y, conf = rest.split(" (")
                device.append((name, float(x), float(y), float(conf.rstrip("%)")) / 100))
        ref = [(["chip", "led"][d["class_index"]], d["x"], d["y"], d["confidence"])
               for d in decode_predictions(g)]
        assert len(device) == len(ref), (run.stdout, ref)
        assert f"{len(ref)} object(s)" in run.stdout
        for dv, rf in zip(device, ref):
            assert dv[0] == rf[0]
            assert np.allclose(dv[1:], rf[1:], atol=2e-3), (dv, rf)
