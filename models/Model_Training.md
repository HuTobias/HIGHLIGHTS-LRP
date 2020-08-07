# Model Training

For training the agents we used the following commit of the OpenAi
baselines repository:
https://github.com/openai/baselines/commits/9ee399f5b20cd70ac0a871927a6cf043b478193f

To get different rewards for each agent (as described in the paper section 4.1) we
modified the *ClipRewardEnv* class of *baselines/commons/atari_wrappers.py*.

*For the *Regular agent* we used the ingame reward but scaled it such
that the minimum reward is 1 not 10:
   ```python
   class ClipRewardEnv(gym.RewardWrapper):
       def __init__(self, env):
           gym.RewardWrapper.__init__(self, env)

       def reward(self, reward):
           return reward/10
   ```

*For the fear ghost agent we used the same ClipRewardEnv as for the
*Regular agent* but we also changed the EpisodicLifeEnv to:

   ```python
   class EpisodicLifeEnv(gym.Wrapper):
     .
     .
     .
     if lives < self.lives and lives > 0:
       # for Qbert sometimes we stay in lives == 0 condition for a few
frames
       # so it's important to keep lives > 0, so that we only reset once
       # the environment advertises done.
       done = True
       reward = -100
   ```

*For the *Power pill agent* we used:

   ```python
   class ClipRewardEnv(gym.RewardWrapper):
       def __init__(self, env):
           gym.RewardWrapper.__init__(self, env)

       def reward(self, reward):
       if reward < 50:
         reward = 0
       if reward > 99:
         reward = 0
         return reward/10
   ```
