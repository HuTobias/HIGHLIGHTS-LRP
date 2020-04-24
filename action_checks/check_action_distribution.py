import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import keras
from custom_atari_wrapper import atari_wrapper
import gym

'''
This module generates streams of gameplay
and calculates the distribution of the actions that the agent chose during this stream.
'''

if __name__ == '__main__':
    create_stream = False

    directory = ''
    #We manually change the name for different agents.
    file_name = 'results_power_pill.csv'

    if create_stream:
        np.random.seed(42)

        #model = keras.models.load_model('models/MsPacman_5M_ingame_reward_action_only.h5')
        #model = keras.models.load_model('models/MsPacman_5M_power_pill_action_only.h5')
        model = keras.models.load_model('models/MsPacman_5M_fear_ghost_action_only.h5')

        steps = 10000

        model.summary()

        env = gym.make('MsPacmanNoFrameskip-v4')
        env.reset()
        wrapper = atari_wrapper(env)
        wrapper.reset(noop_max=1)

        action_list = []

        for _ in range(steps):
            if _ < 4:
                action = env.action_space.sample()
            else:
                my_input = np.expand_dims(stacked_frames, axis=0)
                output = model.predict(my_input)  #this output corresponds with the output in baseline if --dueling=False is correctly set for baselines.

                # save model predictions
                action = np.argmax(np.squeeze(output))
                action_list.append(action)

            stacked_frames, observations, reward, done, info = wrapper.step(action)
            env.render()

        data_frame = pd.DataFrame()
        data_frame['action']= action_list

        data_frame.to_csv(os.path.join(directory, file_name))

    if not create_stream:
        data_frame = pd.read_csv(os.path.join(directory,file_name))

    actions = data_frame.action.values
    n, bins, patches = plt.hist(actions)
    plt.show()

