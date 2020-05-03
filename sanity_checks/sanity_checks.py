'''
This module implements sanity checks for saliency maps.
To this end the layers in the model are cascadingly randomized and for each step we create a copy of the model.
Then we create gameplay and saliency map streams for each of those models, using the decisions of the original model,
 such that all models get the same input states.
Finally we compare the generated saliency of all models.
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
import pandas as pd
import keras
from argmax_analyzer import Argmax
from custom_atari_wrapper import atari_wrapper
import gym
from stream_generator import save_raw_data
import seaborn as sns
from image_utils import normalise_image
from Survey_results.evaluation import show_and_save_plt

import tensorflow as tf
import keras.backend as K


def rand_layer(layer, mean = 0, SD = 0.1):
    '''Custom layer randomization for testing purposes.'''
    weights_shape = layer.get_weights()[0].shape
    bias_shape = layer.get_weights()[1].shape
    rand_weights =  np.random.normal(mean,SD,weights_shape)
    rand_bias = np.random.normal(mean,SD,bias_shape)
    layer.set_weights([rand_weights, rand_bias])

def init_layer(layer):
    ''' Re-initializes the given layer with the original initializer to achieve randomization of the layer that is
    within reasonable bounds for that layer.
    :param layer: the layer to be randomized
    :return: nothing, the given layer is randomized
    '''
    session = K.get_session()
    weights_initializer = tf.variables_initializer(layer.weights)
    session.run(weights_initializer)

def copy_model(model):
    '''
    Copies a keras model including the weights
    :param model: the model to be copied
    :return: the new copy of the model
    '''
    model_m1 = keras.models.clone_model(model)
    model_m1.set_weights(model.get_weights())
    return model_m1

def check_models(model1, model):
    ''' checks if two models have the same weights, to make sure that a layer was randomized.'''
    for i in range(1,7):
        if i != 4:
            print('layer ', i)
            print( (model1.get_layer(index=i).get_weights()[0] == model.get_layer(index=i).get_weights()[0]).all() )
            print( (model1.get_layer(index=i).get_weights()[1] == model.get_layer(index=i).get_weights()[1]).all() )

def calc_sim(learned_relevance, random_relevance):
    ''' Helper function to calculate the similarities of two saliency maps (for learned weights and partly random wheights).
    Only works in this code, since the similarity lists are created elsewhere. '''
    #normalizing:
    learned_relevance = normalise_image(learned_relevance)
    random_relevance = normalise_image(random_relevance)
    neg_random_relevance = 1 - random_relevance

    spearman, spearman2 = spearmanr(random_relevance.flatten(), learned_relevance.flatten(), nan_policy='omit')
    test, _ = spearmanr(neg_random_relevance.flatten(), learned_relevance.flatten(), nan_policy='omit')
    spearman = max(spearman, test)

    # ssim_val = ssim(random_relevance,learned_relevance, multichannel=True)
    ssim_val = ssim(random_relevance.flatten(), learned_relevance.flatten())
    test = ssim(neg_random_relevance.flatten(), learned_relevance.flatten())
    ssim_val = max(ssim_val, test)

    random_hog = hog(random_relevance)
    learned_hog = hog(learned_relevance)
    pearson, _ = pearsonr(random_hog, learned_hog)

    neg_random_hog = hog(neg_random_relevance)
    test, _ = pearsonr(neg_random_hog, learned_hog)
    pearson = max(pearson,test)


    pearson_list.append(pearson)
    ssim_list.append(ssim_val)
    spearman_list.append(spearman)


if __name__ == '__main__':
    # if True, the agent plays a new game, otherwise an old stream is used
    # only needed for the first run
    create_stream = False
    # if True, the similarities are recalculated, otherwise old calculated similarities are loaded
    read_csv = True

    steps = 1000

    directory = ''
    save_file_argmax_raw = os.path.join(directory, 'raw_argmax', 'raw_argmax')
    save_file_state = os.path.join(directory, 'state', 'state')
    save_file1 = os.path.join(directory, 'model1', 'raw_argmax')
    save_file2 = os.path.join(directory, 'model2', 'raw_argmax')
    save_file3 = os.path.join(directory, 'model3', 'raw_argmax')
    save_file4 = os.path.join(directory, 'model4', 'raw_argmax')
    save_file5 = os.path.join(directory, 'model5', 'raw_argmax')

    #create empty list to be filled later
    pearson_list = []
    ssim_list = []
    spearman_list = []
    model_list = []
    action_list = []

    if create_stream:
        # generate stream of states, actions, and saliency maps
        np.random.seed(42)

        model = keras.models.load_model('../models/MsPacman_5M_ingame_reward.h5')
        #model = keras.models.load_model('models/MsPacman_5M_power_pill.h5')
        #model = keras.models.load_model('models/MsPacman_5M_fear_ghost.h5')

        model.summary()

        # create analyzer for fully trained model
        analyzer_arg = Argmax(model)

        # create analyzer for model with randomized last layer
        model1 = copy_model(model)
        layer = model1.get_layer(index=6)
        init_layer(layer)
        check_models(model1, model)
        analyzer1 = Argmax(model1, neuron_selection_mode='index')

        # create analyzer for model where the two last layers are randomized
        model2 = copy_model(model1)
        layer = model2.get_layer(index=5)
        init_layer(layer)
        check_models(model2, model1)
        analyzer2 = Argmax(model2, neuron_selection_mode='index')

        # create analyzer for model where the three last layers are randomized
        model3 = copy_model(model2)
        layer = model3.get_layer(index=3)
        init_layer(layer)
        check_models(model3, model2)
        analyzer3 = Argmax(model3, neuron_selection_mode='index')

        # create analyzer for model where the four last layers are randomized
        model4 = copy_model(model3)
        layer = model4.get_layer(index=2)
        init_layer(layer)
        check_models(model4, model3)
        analyzer4 = Argmax(model4, neuron_selection_mode='index')

        # create analyzer for model where all layers are randomized
        model5 = copy_model(model4)
        layer = model5.get_layer(index=1)
        init_layer(layer)
        check_models(model5, model4)
        analyzer5 = Argmax(model5, neuron_selection_mode='index')

        env = gym.make('MsPacmanNoFrameskip-v4')
        env.reset()
        wrapper = atari_wrapper(env)
        wrapper.reset(noop_max=1)

        for _ in range(steps):
            if _ < 4:
                action = env.action_space.sample()
            else:
                my_input = np.expand_dims(stacked_frames, axis=0)
                output = model.predict(my_input)  #this output corresponds with the output in baseline if --dueling=False is correctly set for baselines.

                action = np.argmax(np.squeeze(output))

                # analyze fully trained model
                argmax = analyzer_arg.analyze(my_input)
                argmax = np.squeeze(argmax)
                # save raw saliency map
                save_raw_data(argmax, save_file_argmax_raw, _)
                # save the state
                save_raw_data(my_input,save_file_state, _)

                # create saliency map for model where the last layer is randomized
                argmax1 = np.squeeze(analyzer1.analyze(my_input, neuron_selection= action ))
                # save raw saliency map
                save_raw_data(argmax1, save_file1, _)
                # calculate similarities and append to lists.
                calc_sim(argmax,argmax1)
                # save chosen action and the layer that was randomized in in this instance
                action_list.append(action)
                model_list.append(1)

                # see above but last two layers are randomized.
                argmax2 = np.squeeze(analyzer2.analyze(my_input, neuron_selection=action))
                save_raw_data(argmax2, save_file2, _)
                calc_sim(argmax, argmax2)
                action_list.append(action)
                model_list.append(2)

                # print((argmax1 == argmax2).all())

                # see above but last three layers are randomized.
                argmax3 = np.squeeze(analyzer3.analyze(my_input, neuron_selection=action))
                save_raw_data(argmax3, save_file3, _)
                calc_sim(argmax, argmax3)
                action_list.append(action)
                model_list.append(3)

                # print((argmax3 == argmax2).all())

                # see above but last four layers are randomized.
                argmax4 = np.squeeze(analyzer4.analyze(my_input, neuron_selection=action))
                save_raw_data(argmax4, save_file4, _)
                calc_sim(argmax, argmax4)
                action_list.append(action)
                model_list.append(4)

                # see above but all layers are randomized.
                argmax5 = np.squeeze(analyzer5.analyze(my_input, neuron_selection=action))
                save_raw_data(argmax5, save_file5, _)
                calc_sim(argmax, argmax5)
                action_list.append(action)
                model_list.append(5)



            stacked_frames, observations, reward, done, info = wrapper.step(action)
            env.render()

        data_frame = pd.DataFrame(columns=['rand_layer', 'pearson', 'ssim', 'spearman', 'action'])
        data_frame['rand_layer'] = model_list
        data_frame['pearson'] = pearson_list
        data_frame['ssim'] = ssim_list
        data_frame['spearman'] = spearman_list
        data_frame['action'] = action_list

        data_frame.to_csv('results.csv')

    else:
        if read_csv:
            data_frame = pd.read_csv('results.csv')
        else:
            # recalculate similarities
            for i in range(4,1000):
                index = '_' + str(i) + '.npy'
                raw_argmax = np.load(save_file_argmax_raw + index)

                argmax1 = np.load(save_file1 + index)
                calc_sim(raw_argmax,argmax1)
                model_list.append(1)

                argmax2 = np.load(save_file2 + index)
                calc_sim(raw_argmax, argmax2)
                model_list.append(2)

                calc_sim(raw_argmax, np.load(save_file3 + index))
                model_list.append(3)

                calc_sim(raw_argmax, np.load(save_file4 + index))
                model_list.append(4)

                calc_sim(raw_argmax, np.load(save_file5 + index))
                model_list.append(5)

                data_frame = pd.DataFrame(columns=['rand_layer', 'pearson', 'ssim', 'spearman', 'action'])
                data_frame['rand_layer'] = model_list
                data_frame['pearson'] = pearson_list
                data_frame['ssim'] = ssim_list
                data_frame['spearman'] = spearman_list

                data_frame.to_csv('results.csv')

    #Create plots
    data_frame['rand_layer'] = data_frame.rand_layer.apply(
        lambda x: 'fc_2' if x == 1 else 'fc_1' if x == 2 else 'conv_3' if x == 3 else 'conv_2' if x == 4 else 'conv_1')

    sns.set(palette='colorblind', style="whitegrid")

    ax = sns.barplot(x='rand_layer', y='pearson', data=data_frame)
    show_and_save_plt(ax, 'pearson',label_size=28, tick_size=20, y_label='Pearson')
    ax = sns.barplot(x='rand_layer', y='ssim', data=data_frame)
    plt.xlabel(None)
    show_and_save_plt(ax, 'ssim',label_size=28, tick_size=20, y_label='Ssim')
    ax = sns.barplot(x='rand_layer', y='spearman', data=data_frame)
    plt.xlabel(None)
    show_and_save_plt(ax, 'spearman',label_size=28, tick_size=20, y_label='Spearman')

