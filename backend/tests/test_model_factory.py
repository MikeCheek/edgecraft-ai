import numpy as np
from tensorflow import keras

from app.services.model_factory import set_backbone_trainable


def _model_with_backbone():
    inp = keras.Input((8, 8, 3))
    h = keras.layers.Conv2D(4, 3, padding="same")(inp)
    h = keras.layers.BatchNormalization()(h)
    backbone = keras.Model(inp, keras.layers.ReLU()(h), name="backbone")
    backbone.trainable = False
    x = keras.Input((8, 8, 3))
    y = keras.layers.GlobalAveragePooling2D()(backbone(x, training=False))
    return keras.Model(x, keras.layers.Dense(2)(y)), backbone


def test_unfrozen_backbone_keeps_batchnorm_in_inference_mode():
    """Regression: Keras 3 drops the construction-time training=False once
    the backbone is trainable, so fine-tuning ran pretrained BatchNorm in
    training mode and collapsed train accuracy."""
    model, backbone = _model_with_backbone()
    bn = next(layer for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization))
    bn.moving_mean.assign(np.full(4, -3.0, "float32"))  # stats unlike any single batch
    x = np.random.default_rng(0).random((4, 8, 8, 3)).astype("float32")

    set_backbone_trainable(backbone)
    np.testing.assert_allclose(model(x, training=True), model(x, training=False), atol=1e-5)
    assert any(w.trainable for layer in backbone.layers if not isinstance(layer, keras.layers.BatchNormalization)
               for w in layer.weights)

    bn.trainable = True  # the old behaviour: a trainable BN normalises with batch statistics
    assert not np.allclose(model(x, training=True), model(x, training=False), atol=1e-5)


def test_trainable_layers_limits_unfreezing():
    _, backbone = _model_with_backbone()
    set_backbone_trainable(backbone, trainable_layers=1)
    assert [layer.trainable for layer in backbone.layers][-1] is True
    assert not any(layer.trainable for layer in backbone.layers[:-1])
