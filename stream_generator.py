import gym
import matplotlib.pyplot as plt
from custom_atari_wrapper import atari_wrapper
import numpy as np
import keras
import innvestigate
import os
import image_utils
from argmax_analyzer import Argmax

def save_frame(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
    plt.imsave(save_file + '_' + str(frame) + '.png', array)

def save_array(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
    np.save(save_file + '_' + str(frame) + '.npy', array)

def save_q_values(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
    save_file = save_file + '_' + str(frame) + '.txt'
    with open(save_file, "w") as text_file:
        text_file.write(str(array))

#model = keras.models.load_model('models/MsPacman_500K_reward26komma9_action_only.h5')
#model = keras.models.load_model('models/MsPacman_1M_reward36komma3_action_only.h5')
model = keras.models.load_model('models/MsPacman_2M_2_reward47komma5_action_only.h5')


model.summary()

#analyzer_z = innvestigate.analyzer.LRPAlpha1Beta0IgnoreBias(model)
analyzer_arg = Argmax(model)

env = gym.make('MsPacmanNoFrameskip-v4')
env.reset()
wrapper = atari_wrapper(env)

save_file_argmax = os.path.join('stream', 'argmax', 'argmax')
save_file_z = os.path.join('stream', 'z_rule', 'z_rule')
save_file_screen = os.path.join('stream', 'screen', 'screen')
save_file_state = os.path.join('stream', 'state', 'state')
save_file_q_values = os.path.join('stream', 'q_values', 'q_values')

if __name__ == '__main__':
    for _ in range(10000):
        if _ < 4:
            action = env.action_space.sample()
        else:
            my_input = np.expand_dims(stacked_frames, axis=0)
            output = model.predict(my_input)
            action = np.argmax(np.squeeze(output))

            #analyzing
            argmax = analyzer_arg.analyze(my_input)
            argmax = np.squeeze(argmax)
            argmax = image_utils.normalise_image(argmax)

            # for future work
            # z_rule = analyzer_z.analyze(my_input)
            # z_rule = np.squeeze(z_rule)
            # z_rule = image_utils.normalise_image(z_rule)

            save_array(my_input,save_file_state, _)
            image = np.squeeze(my_input)
            image = np.hstack((image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3]))
            save_frame(image,save_file_state, _)

            save_q_values(output,save_file_q_values, _)


            for i in range(len(observations)):
                index = str(_) + '_' + str(i)
                observation = observations[i]
                save_frame(observation, save_file_screen, index)
                saliency = image_utils.add_saliency_to_image(argmax[:, :, 3], observation, 2)
                save_frame(saliency, save_file_argmax, index)

                # # for future work
                # saliency = image_utils.add_saliency_to_image(z_rule[:, :, 3], observation, 2)
                # save_frame(saliency, save_file_z, index)



        stacked_frames, observations, reward, done, info = wrapper.step(action)
        env.render()


