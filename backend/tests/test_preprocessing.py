import numpy as np

from app.services import preprocessing as P
from tests.conftest import make_png, make_wav


def test_mfcc_matches_librosa():
    import librosa

    y = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    ours = P.compute_mfcc(y, "KEYWORD_SPOTTING")
    S = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=320, center=False,
                                       n_mels=40, fmin=20, fmax=8000, power=2.0)
    ref = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=np.max, top_db=80), n_mfcc=40)
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    assert ours.shape == (40, 49)
    assert np.abs(ours - ref).max() < 1e-4


def test_mfcc_is_gain_invariant():
    y = np.random.default_rng(1).standard_normal(16000).astype(np.float32) * 0.1
    a = P.compute_mfcc(y, "KEYWORD_SPOTTING")
    b = P.compute_mfcc(y * 7.5, "KEYWORD_SPOTTING")
    assert np.abs(a - b).max() < 1e-3


def test_audio_features_shape_and_padding():
    x = P.preprocess(make_wav(seconds=0.4), "KEYWORD_SPOTTING", (40, 49, 1))
    assert x.shape == (40, 49, 1)
    x = P.preprocess(make_wav(seconds=1.0), "KEYWORD_SPOTTING", (13, 60, 1))
    assert x.shape == (13, 60, 1)
    assert np.all(x[:, 49:, 0] == 0)  # padded frames


def test_image_preprocessing_range_and_channels():
    rgb = P.preprocess(make_png(), "IMAGE_CLASSIFICATION", (24, 40, 3))
    gray = P.preprocess(make_png(), "VISUAL_WAKE_WORDS", (24, 40, 1))
    assert rgb.shape == (24, 40, 3) and gray.shape == (24, 40, 1)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0


def test_quantize_input_rounds_and_saturates():
    q = P.quantize_input(np.array([1.0, 0.999, -5.0, 9.0], np.float32), 1 / 255.0, -128)
    assert q.tolist() == [127, 127, -128, 127]


def test_default_shapes():
    assert P.default_input_shape("KEYWORD_SPOTTING") == (40, 49, 1)
    assert P.default_input_shape("AUDIO_CLASSIFICATION") == (40, 99, 1)
    assert P.default_input_shape("VISUAL_WAKE_WORDS") == (96, 96, 1)
