import gym
import cv2
import numpy as np


class atari_wrapper():
    ''' simple implementation of openai's atari_wrappers for our purposes'''
    def __init__(self, env):
        self.env = env
        self.width = 84
        self.height = 84
        self.zeros = np.zeros((self.width,self.height))
        self.stacked_frame = np.stack((self.zeros,self.zeros,self.zeros,self.zeros), axis=-1)
        self.noop_action = 0

    def preprocess_frame(self,frame):
        ''' preprocessing according to openai's atari_wrappers.WrapFrame
            also applys scaling between 0 and 1 which is done in tensorflow in baselines
        :param frame: the input frame
        :return: rescaled and greyscaled frame
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame / 255
        return frame[:, :, None]

    def update_stacked_frame(self,new_frame):
        ''' adds the new_frame to the stack of 4 frames, while shifting each other frame one to the left
        :param new_frame:
        :return: the new stacked frame
        '''
        for i in range(3):
            self.stacked_frame[:,:,i] = self.stacked_frame[:,:,i+1]
        new_frame = self.preprocess_frame(new_frame)
        new_frame = np.squeeze(new_frame)
        self.stacked_frame[:,:,3] = new_frame
        return self.stacked_frame

    def step(self,action, skip_frames=4):
        max_frame, stacked_observations, reward, done, info = self.repeat_frames(action, skip_frames=skip_frames)
        stacked_frames = self.update_stacked_frame(max_frame)
        #reset the environment if the game ended
        if done:
            self.reset()
        return stacked_frames, stacked_observations, reward, done, info

    def repeat_frames(self, action, skip_frames=4):
        ''' skip frames to be inline with baselines DQN. stops when the current game is done
        :param action: the choosen action which will be repeated
        :param skip_frames: the number of frames to skip
        :return max frame: the frame used by the agent
        :return stacked_observations: all skipped observations '''
        stacked_observations = []
        #TODO dirty numbers
        obs_buffer = np.zeros((2,210,160,3),dtype='uint8' )
        for i in range(skip_frames):
            observation, reward, done, info = self.env.step(action)
            if i == skip_frames - 2: obs_buffer[0] = observation
            if i == skip_frames - 1: obs_buffer[1] = observation
            if done:
                break
            stacked_observations.append(observation)

        max_frame = obs_buffer.max(axis=0)
        return max_frame, stacked_observations, reward, done, info

    def reset(self, noop_max = 30):
        """ Do no-op action for a number of steps in [1, noop_max], to achieve random game starts.
        """
        self.env.reset()
        noops = np.random.randint(1, noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs

    def fixed_reset(self, step_number, action):
        '''
        Create a fixed starting position for the environment by doing *action* for *step_number* steps
        :param step_number: number of steps to be done at the beginning of the game
        :param action: action to be done at the start of the game
        :return: obs at the end of the starting sequence
        '''
        self.env.reset()
        for _ in range(step_number):
            obs, _, done, _ = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
        return obs