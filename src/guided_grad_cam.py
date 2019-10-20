import sys
import numpy as np

from keras.models import load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf

import cv2

# Guided_backprop
def register_gradient():
    '''
    1. set negative gradients to 0
    2. set the gradients of which the input is negative to 0
    '''
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

def modify_backprop(model_path, name="GuidedBackProp"):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
        new_model = load_model(model_path)
    return new_model

def compile_saliency_function(model, activation_layer='block4_conv1'):
    layer_output = model.get_layer(activation_layer).output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), model.input)[0]
    return K.function([model.input, K.learning_phase()], [saliency])

# Grad-CAM
