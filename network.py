import numpy as np
from scipy.spatial.distance import cosine

import keras
import keras.backend as kb
import keras.layers as kl
import keras.initializers as kint
import keras.activations as kact
from keras.engine.topology import InputSpec
from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, f1_score

class HingeLoss:

    def __init__(self, margin, negative_weight=1.0):
        self.margin = margin
        self.negative_weight = negative_weight

    def __call__(self, y_true, y_pred):
        y_true_ = 2 * y_true - 1
        FN_weight = kb.ones_like(y_true_, dtype="float32") * self.negative_weight
        FP_weight = kb.ones_like(y_true_, dtype="float32")
        weights = kb.switch(y_true_ > 0, FN_weight, FP_weight)
        result = kb.maximum(weights * (self.margin - y_true_ * y_pred), 0)
        return result


def sign_accuracy(y_true, y_pred):
    return kb.mean(kb.equal(y_true, kb.cast(kb.greater_equal(y_pred, 0), y_true.dtype)))


def cosine_similarity(a, b):
    a_norm = kb.sqrt(kb.sum(a * a, axis=-1))
    b_norm = kb.sqrt(kb.sum(b * b, axis=-1))
    a /= a_norm[:,None]
    b /= b_norm[:,None]
    return (1.0 - kb.sum(a * b, axis=-1))


class SimilarityLayer(kl.Layer):

    def __init__(self, layers, use_svo=False, activation=None, initial_threshold=None, **kwargs):
        self.layers = layers
        self.use_svo = use_svo
        self.activation = kact.get(activation)
        self.gain = 1.0
        self.initial_threshold = initial_threshold
        super(SimilarityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        first_shape, second_shape = input_shape
        assert first_shape == second_shape
        assert len(first_shape) == 2
        input_dim = first_shape[-1]
        dim = (input_dim // 3) if self.use_svo else input_dim

        self.kernels = [None] * self.layers
        self.biases = [None] * self.layers
        if self.layers >= 1:
            self.kernels[0] = self.add_weight(
                shape=(input_dim, dim), initializer=kint.Zeros(), name="similarity_kernel_1")
            self.biases[0] = self.add_weight(
                shape=(dim,), initializer=kint.Zeros(), name="similarity_bias_0")
        for i in range(1, self.layers):
            self.kernels[i] = self.add_weight(shape=(dim, dim), initializer=kint.Identity(gain=self.gain),
                                              name="similarity_kernel_{}".format(i+1))
            self.biases[i] = self.add_weight(
                shape=(dim,), initializer=kint.Zeros(), name="similarity_bias_{}".format(i+1))
        self.threshold = self.add_weight(
            name="similarity_threshold", shape=(1,), initializer=kint.Constant(self.initial_threshold))
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: input_dim})] * 2
        self.built = True
        super(SimilarityLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = inputs[:]
        for i, (kernel, bias) in enumerate(zip(self.kernels, self.biases)):
            inputs[0] = self.activation(kb.bias_add(kb.dot(inputs[0], kernel), bias))
            inputs[1] = self.activation(kb.bias_add(kb.dot(inputs[1], kernel), bias))
        similarity = cosine_similarity(inputs[0], inputs[1])[:,None]  # to make target 2D
        return self.threshold - similarity

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


def set_initial_model_weights(model, use_svo, dim):
    layer = model.get_layer(index=2)
    if use_svo:
        kernel = np.vstack([np.eye(dim, dtype=float) for _ in range(3)])
    else:
        kernel = np.eye(dim, dtype=float)
    curr_weights = layer.get_weights()
    curr_weights[0] = kernel
    layer.set_weights(curr_weights)
    return

def build_network(dim, layers=2, use_svo=False, activation=None,
                  dropout=0.0, initial_threshold=0.5, false_negative_weight=1.0,
                  margin=0.1):
    input_dim = (dim * 3) if use_svo else dim
    first, second = kl.Input(shape=(input_dim,)), kl.Input(shape=(input_dim,))
    inputs = [first, second]

    if dropout > 0.0:
        first, second = kl.Dropout(dropout)(first), kl.Dropout(dropout)(second)

    score = SimilarityLayer(layers=layers, use_svo=use_svo, activation=activation,
                            initial_threshold=initial_threshold)([first, second])

    model = keras.Model(inputs, score)
    loss = HingeLoss(margin, negative_weight=false_negative_weight)
    model.compile(optimizer="adam", loss=loss, metrics=[sign_accuracy])
    set_initial_model_weights(model, use_svo, dim)
    print(model.summary())
    return model

# build_network(100, use_svo=True, false_negative_weight=1.25)