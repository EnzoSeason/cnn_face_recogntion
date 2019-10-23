import sys
import numpy as np

from keras.models import load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf

from skimage import io, util, color, transform

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
def calc_importance(model, preprocessed_input, layer_name, class_index):
    # get class score (input of the softmax)
    yc = model.output.op.inputs[0][0, class_index]
    # get feature maps(output of the given layer)
    Ak = model.get_layer(layer_name).output
    
    # calculate the gradients
    print('Run Grad_CAM on class '+str(class_index)+', layer '+layer_name)

    # define the gradients function
    grads = K.gradients(yc, [Ak])[0]
    grads_func = K.function(inputs=[model.input], outputs=[Ak, grads])
    # calculation
    Ak_val, grads_val = grads_func([preprocessed_input])
    Ak_val, grads_val = Ak_val.squeeze(), grads_val.squeeze()
    
    # calculate the importance
    importance = np.mean(grads_val, axis=(0,1))

    return(importance, Ak_val)

def calc_gradcam(preprocessed_input, importance, features_maps_val):
    # create heatmap
    heatmap = np.maximum(0, np.sum(importance*features_maps_val, axis=2))
    # create output
    output_image = heatmap * preprocessed_input

    return (heatmap, output_image)
 
