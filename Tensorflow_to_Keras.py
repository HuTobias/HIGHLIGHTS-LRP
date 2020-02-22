import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

import joblib
import os
import numpy as np

model_path = './models'
model_name = "MsPacman_5M_power_pill"
load_path = os.path.join(model_path,model_name)

def load_Model_with_trained_variables(load_path):
    # Load Checkpoint
    tf.reset_default_graph()
    dictOfWeights = {}; dictOfBiases = {}
    with tf.Session() as sess:
        col = joblib.load(os.path.expanduser(load_path))
        i = 0
        for var in col:
            i = i+1
            if type(var) is np.ndarray:
                print(str(i) + " " + var + str(col[var].shape))
            else:
                print(str(i) + " " + var + " no ndarray")
            if "target" not in var:
                if "weights" in var:
                    dictOfWeights[var] = col[var]
                if "biases" in var:
                    dictOfBiases[var] = col[var]
            pass

    # Keras Model
    hidden = 256
    ThirdLastBiases = dictOfBiases['deepq/q_func/action_value/fully_connected_1/biases:0']
    num_actions = ThirdLastBiases.size
    dueling = True
    inputs = Input(shape=(84, 84, 4))
    # ConvLayer
    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='SAME', name='deepq/q_func/convnet/Conv')(inputs)
    x1 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='SAME', name='deepq/q_func/convnet/Conv_1')(x)
    x2 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='SAME', name='deepq/q_func/convnet/Conv_2')(x1)
    conv_out = Flatten()(x2)
    # Action stream
    action_out = conv_out
    action_out = Dense(hidden, activation='relu', name='deepq/q_func/action_value/fully_connected')(action_out)  # hidden=512 (paper), 256 (train_pong)
    action_scores = Dense(num_actions, name='deepq/q_func/action_value/fully_connected_1', activation='linear')(action_out)  # num_actions = Anzahl valide Aktion zw. {4, .., 18}
    # State stream
    if dueling:
        state_out = conv_out
        state_out = Dense(hidden, activation='relu', name='deepq/q_func/state_value/fully_connected')(state_out)  # hidden=512 (paper), 256 (train_pong)
        state_score = Dense(1, name='deepq/q_func/state_value/fully_connected_1')(state_out)
    # Finish model
    model = Model(inputs, [action_scores, state_score])
    modelActionPart = Model(inputs, action_scores)
    modelStatePart  = Model(inputs, state_score)

    # Load weights
    for layer in model.layers:
        # if not same sizes: Layer weight shape (3136, 256) not compatible with provided weight shape (7744, 256)
        if layer.name + "/weights:0" in dictOfWeights:
            newWeights = dictOfWeights[layer.name + "/weights:0"]
            newBiases = dictOfBiases[layer.name + "/biases:0"]
            # set_weights(list of ndarrays with 2 elements: 0->weights 1->biases)
            layer.set_weights([newWeights, newBiases])
            print("Found and applied values for layer " + layer.name)
        else:
            print("No corresponding layer found for" + layer.name)
    return model, modelStatePart, modelActionPart

if __name__ == '__main__':
    (model, modelStatePart, modelActionPart) = load_Model_with_trained_variables(load_path)
    model.summary()
    modelStatePart.summary()
    modelActionPart.summary()

    #model.save(load_path +'_dueling.h5')
    #modelStatePart.save(load_path + '_state_only.h5')
    modelActionPart.save(load_path + '_action_only.h5')

pass