import numpy as np
from scipy.spatial.distance import cosine

import keras
import keras.backend as kb
import keras.layers as kl
import keras.initializers as kint
import keras.activations as kact
import keras.regularizers as kreg
from keras.engine.topology import InputSpec
from keras.callbacks import EarlyStopping

import tensorflow as tf

class RandomizedIdentityInitializer:

    def __init__(self, dim, output_dim=None, identity_dim=None,
                 basic_initializer="identity", noise=0.0):
        self.dim = dim
        self.output_dim = output_dim or dim
        self.identity_dim = identity_dim or dim
        if basic_initializer == "identity" and (self.dim % self.identity_dim != 0):
            raise ValueError("Identity dim must divide input dim")
        if self.output_dim < self.identity_dim:
            raise ValueError("Output dim must be at least equal to identity dim")
        self.mult = self.dim // self.identity_dim
        self.basic_initializer = basic_initializer
        self.noise = noise

    def _make_basic_kernel(self, dtype):
        if self.basic_initializer == "identity":
            kernels = [kb.eye(self.identity_dim, dtype=dtype) for _ in range(self.mult)]
            kernel = kb.concatenate(kernels, axis=0)
        elif self.basic_initializer == "ones":
            kernel = kb.ones(shape=(self.dim, self.identity_dim), dtype=dtype)
        else:
            kernel = kb.zeros(shape=(self.dim, self.identity_dim), dtype=dtype)
        if self.output_dim > self.identity_dim:
            other_kernel = kb.zeros(shape=(self.dim, self.output_dim - self.identity_dim), dtype=dtype)
            kernel = kb.concatenate([kernel, other_kernel])
        return kernel

    def __call__(self, shape, dtype=None):
        assert shape[0] == self.dim and shape[1] == self.output_dim
        dtype = dtype or "float32"
        kernel = self._make_basic_kernel(dtype=dtype)
        if self.noise > 0.0:
            kernel += kb.random_uniform(shape=shape, minval=-self.noise, maxval=self.noise)
        return kernel


class IdentityRegularizer:

    def __init__(self, l2, dim, use_svd=False):
        self.l2 = l2
        self.dim = dim
        self.use_svd = use_svd

    def __call__(self, a):
        if self.use_svd:
            v, _, _ = tf.svd(a)
            return self.l2 * kb.square(kb.maximum(v[0], 1) - 1)
        else:
            return kb.sum(self.l2 * kb.square(a - kb.eye(self.dim)))



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

    def __init__(self, layers, use_svo=False, activation=None,
                 use_bilinear_similarity=False,
                 output_dim=None, initial_threshold=0.25,
                 noise=0.0, regularizer=0.0, use_svd_regularizer=False,
                 **kwargs):
        self.layers = layers
        self.use_svo = use_svo
        self.activation = kact.get(activation)
        self.use_bilinear_similarity = use_bilinear_similarity
        self.output_dim = output_dim
        self.initial_threshold = initial_threshold
        self.noise = noise
        self.regularizer = regularizer
        self.use_svd_regularizer = use_svd_regularizer
        self.gain = 1.0
        super(SimilarityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        first_shape, second_shape = input_shape
        assert first_shape == second_shape
        assert len(first_shape) == 2
        input_dim = first_shape[-1]
        identity_dim = (input_dim // 3) if self.use_svo else input_dim
        dim = self.output_dim or identity_dim

        self.kernels = [None] * self.layers
        self.biases = [None] * self.layers
        first_regularizer, second_regularizer = None, None
        if self.regularizer > 0.0:
            if dim == input_dim:
                first_regularizer = IdentityRegularizer(self.regularizer, dim,
                                                        self.use_svd_regularizer)
            else:
                Warning("You cannot use IdentityRegularizer with non-square matrices")
        if self.layers >= 1:
            kernel_initializer = RandomizedIdentityInitializer(input_dim, dim, identity_dim, noise=self.noise)
            self.kernels[0] = self.add_weight(shape=(input_dim, dim),
                                              initializer=kernel_initializer,
                                              regularizer=first_regularizer,
                                              name="similarity_kernel_1")
            self.biases[0] = self.add_weight(
                shape=(dim,), initializer=kint.Zeros(), name="similarity_bias_0")
        if self.regularizer > 0.0:
            second_regularizer = IdentityRegularizer(self.regularizer, dim, self.use_svd_regularizer)
        for i in range(1, self.layers):
            kernel_initializer = RandomizedIdentityInitializer(dim, noise=self.noise)
            self.kernels[i] = self.add_weight(shape=(dim, dim),
                                              initializer=kernel_initializer,
                                              regularizer=second_regularizer,
                                              name="similarity_kernel_{}".format(i+1))
            self.biases[i] = self.add_weight(
                shape=(dim,), initializer=kint.Zeros(), name="similarity_bias_{}".format(i+1))
        if self.use_bilinear_similarity:
            similarity_initializer = RandomizedIdentityInitializer(dim, noise=self.noise)
            self.similarity_matrix = self.add_weight(
                shape=(dim, dim), initializer=similarity_initializer,
                name="similarity_matrix")
        self.threshold = self.add_weight(name="similarity_threshold", shape=(1,),
                                         initializer=kint.Constant(self.initial_threshold))
        self.input_spec = [InputSpec(min_ndim=2, axes={-1: input_dim})] * 2
        self.built = True
        super(SimilarityLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = inputs[:]
        for i, (kernel, bias) in enumerate(zip(self.kernels, self.biases)):
            inputs[0] = self.activation(kb.bias_add(kb.dot(inputs[0], kernel), bias))
            inputs[1] = self.activation(kb.bias_add(kb.dot(inputs[1], kernel), bias))
        if self.use_bilinear_similarity:
            inputs[0] /= kb.sqrt(kb.sum(inputs[0] * inputs[0], axis=1))[:,None]
            inputs[1] /= kb.sqrt(kb.sum(inputs[1] * inputs[1], axis=1))[:,None]
            first = kb.dot(inputs[0], self.similarity_matrix)
            similarity = 1.0 - kb.sum(first * inputs[1], axis=1)[:,None]
        else:
            similarity = cosine_similarity(inputs[0], inputs[1])[:,None]  # to make target 2D
        return self.threshold - similarity

    def compute_output_shape(self, input_shape):
        return [(None, 1)]


class NetworkBuilder:

    def __init__(self, dim, use_svo=False, input_dim=None, dropout=0.0,
                 false_negative_weight=1.0, margin=0.1, **kwargs):
        self.dim = dim
        self.use_svo = use_svo
        self.input_dim = input_dim or ((self.dim * 3) if self.use_svo else self.dim)
        self.dropout = dropout
        self.false_negative_weight = false_negative_weight
        self.margin = margin
        self.kwargs = kwargs

    def build(self):
        first, second = kl.Input(shape=(self.input_dim,)), kl.Input(shape=(self.input_dim,))
        inputs = [first, second]

        if self.dropout > 0.0:
            first, second = kl.Dropout(self.dropout)(first), kl.Dropout(self.dropout)(second)

        score = SimilarityLayer(use_svo=self.use_svo, output_dim=self.dim, **self.kwargs)([first, second])

        self.model = keras.Model(inputs, score)
        loss = HingeLoss(self.margin, negative_weight=self.false_negative_weight)
        self.model.compile(optimizer="adam", loss=loss, metrics=[sign_accuracy])
        print(self.model.summary())
        return self.model

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