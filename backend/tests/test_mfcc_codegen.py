"""The generated on-device MFCC front-end must match the training one."""

import shutil
import subprocess

import numpy as np
import pytest

from app.services import preprocessing
from app.utils.mfcc_codegen import generate_mfcc_header

HARNESS = r"""
#include <stdio.h>
#include "mfcc_frontend.h"
static float audio[MFCC_N_SAMPLES];
static float out[MFCC_N_MFCC * MFCC_OUT_FRAMES];
int main() {
  for (int i = 0; i < MFCC_N_SAMPLES; i++) if (scanf("%f", &audio[i]) != 1) return 1;
  mfcc_compute(audio, out);
  for (int i = 0; i < MFCC_N_MFCC * MFCC_OUT_FRAMES; i++) printf("%.7g\n", out[i]);
  return 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("task,n_mfcc,frames", [("KEYWORD_SPOTTING", 40, 49), ("AUDIO_CLASSIFICATION", 13, 120)])
def test_generated_mfcc_matches_python(tmp_path, task, n_mfcc, frames):
    (tmp_path / "mfcc_frontend.h").write_text(generate_mfcc_header(task, n_mfcc, frames))
    (tmp_path / "main.cpp").write_text(HARNESS)
    subprocess.run(["g++", "-O2", "-o", str(tmp_path / "mfcc"), str(tmp_path / "main.cpp")], check=True)

    p = preprocessing.audio_params(task, n_mfcc)
    rng = np.random.default_rng(1)
    t = np.arange(int(p["n_samples"])) / 16000.0
    audio = (0.3 * np.sin(2 * np.pi * 700 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)

    run = subprocess.run([str(tmp_path / "mfcc")], input="\n".join(map(str, audio)),
                         capture_output=True, text=True, check=True)
    device = np.array(run.stdout.split(), dtype=np.float32).reshape(n_mfcc, frames)

    ref = preprocessing.compute_mfcc(audio, task, n_mfcc)
    ref = ref[:, :frames] if ref.shape[1] >= frames else np.pad(ref, ((0, 0), (0, frames - ref.shape[1])))
    assert np.abs(device - ref).max() < 2e-3
