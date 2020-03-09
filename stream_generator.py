import gym
import matplotlib.pyplot as plt
from custom_atari_wrapper import atari_wrapper
import numpy as np
import keras
import innvestigate
from argmax_analyzer import Argmax
import overlay_stream

#Quickfix for argmax
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def save_frame(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
        os.rmdir(save_file)
    plt.imsave(save_file + '_' + str(frame) + '.png', array)

def save_array(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
        os.rmdir(save_file)
    np.save(save_file + '_' + str(frame) + '.npy', array)

def save_q_values(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
        os.rmdir(save_file)
    save_file = save_file + '_' + str(frame) + '.txt'
    with open(save_file, "w") as text_file:
        text_file.write(str(array))

def save_raw_data(array,save_file, frame):
    '''
    saves a raw state or saliency map as array and as image
    :param array: array to be saved
    :param save_file: file path were the data should be saved
    :param frame: the frame index of the file
    :return: None
    '''
    save_array(array,save_file, frame)
    image = np.squeeze(array)
    image = np.hstack((image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]))
    save_frame(image, save_file, frame)

def get_feature_vector(model, input):
    '''
    returns the output of the second to last layer, which act similar to a feature vector for the DQN-network
    :param model: the model used for prediction
    :param input: the input for the prediction
    :return:
    '''
    helper_func = keras.backend.function([model.layers[0].input],
                                  [model.layers[-2].output])
    features = helper_func([input])[0]
    features = np.squeeze(features)
    return features

if __name__ == '__main__':
    #use a different start to get states outside of the highlights stream
    fixed_start = False

    np.random.seed(42)

    model = keras.models.load_model('models/MsPacman_5M_ingame_reward.h5')
    #model = keras.models.load_model('models/MsPacman_5M_power_pill.h5')
    #model = keras.models.load_model('models/MsPacman_5M_fear_ghost.h5')

    steps = 10000

    model.summary()

    analyzer_arg = Argmax(model)

    total_reward = 0
    reward_list = []

    env = gym.make('MsPacmanNoFrameskip-v4')
    env.reset()
    wrapper = atari_wrapper(env)
    wrapper.reset(noop_max=1)
    if fixed_start :
        wrapper.fixed_reset(300,2) #used  action 3 and 4

    directory = ''
    directory = os.path.join(directory,'stream')
    save_file_argmax = os.path.join(directory, 'argmax', 'argmax')
    save_file_argmax_raw = os.path.join(directory, 'raw_argmax', 'raw_argmax')
    save_file_z = os.path.join(directory, 'z_rule', 'z_rule')
    save_file_screen = os.path.join(directory, 'screen', 'screen')
    save_file_state = os.path.join(directory, 'state', 'state')
    save_file_q_values = os.path.join(directory, 'q_values', 'q_values')
    save_file_features = os.path.join(directory, 'features', 'features')
    scores_file = os.path.join(directory, 'scores.txt')
    average_score_file = os.path.join(directory, 'average_score.txt')

    for _ in range(steps):
        if _ < 4:
            action = env.action_space.sample()
            # to have more controll over the fixed starts
            if fixed_start:
                action=0
        else:
            my_input = np.expand_dims(stacked_frames, axis=0)
            output = model.predict(my_input)  #this output corresponds with the output in baseline if --dueling=False is correctly set for baselines.
            # save model predictions
            save_q_values(output, save_file_q_values, _)
            features = get_feature_vector(model, my_input)
            save_q_values(features,save_file_features,_)
            save_array(features, save_file_features,_)

            action = np.argmax(np.squeeze(output))

            ###action saving
            # if action != old_action:
            #     print('action', action, 'state', _)
            # old_action = action


            #analyzing
            argmax = analyzer_arg.analyze(my_input)
            argmax = np.squeeze(argmax)
            # save raw saliency
            save_raw_data(argmax, save_file_argmax_raw, _)

            #save the state
            save_raw_data(my_input,save_file_state, _)

            #save screen output, and screen + saliency
            for i in range(len(observations)):
                index = str(_) + '_' + str(i)
                observation = observations[i]
                save_frame(observation, save_file_screen, index)


        stacked_frames, observations, reward, done, info = wrapper.step(action)
        total_reward += reward
        if done:
            print('total_reward',total_reward)
            reward_list.append(total_reward)
            total_reward = 0
        env.render()

    reward_list.append(total_reward)
    average_reward = np.mean(reward_list)
    with open(scores_file, "w") as text_file:
        text_file.write(str(reward_list))
    with open(average_score_file, "w") as text_file:
        text_file.write(str(average_reward))

    import datetime
    print('Time:')
    print(datetime.datetime.now())

    overlay_stream.overlay_stream(directory)