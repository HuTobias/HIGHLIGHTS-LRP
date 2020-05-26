# Local and Global Explanations of Agent Behavior:Integrating Strategy Summaries with Saliency Map

This repository contains the implementation for the paper "Local and Global Explanations of Agent Behavior:Integrating Strategy Summaries with Saliency Map"(https://arxiv.org/abs/2005.08874).
This paper combines global explanations in the form of HIGHLIGHTS-DIV policy summaries (https://dl.acm.org/doi/10.5555/3237383.3237869) with LRP-argmax salieny maps (https://www.springerprofessional.de/enhancing-explainability-of-deep-reinforcement-learning-through-/17150184) 
by generating summaries of Atari agent behavior that is overlayd with saliency maps that show what information the agent used.

# Installation

We only tested with python 3.6.5.
It should be enough to install the given requirements.
For gym to work on a windows system you have to follow the instructions in *gym_for_windows.txt*.

*install_argmax.bat* is not neccessary anymore but can be used to update the argmax analyzer should the coresponding repository change.

# Summary Creation
The models in the folder *models* were trained with the openai-baselines repository https://github.com/openai/baselines.

*Tensorflow_to_Keras.py* converts the original tensorflow models to keras models.
Then *stream_generator.py* creates a stream of gameplay, saving all states, visual frames, Q-values and raw LRP-argmax saliency maps (generated with *argmax_analyzer.py* from https://github.com/HuTobias/LRP_argmax). 
At the very end of *stream_generator.py*, *overlay_stream.py* is used to overlay each frame with a saliency map.
This can also be redone later using *overlay_stream.py* to save time while trying different overlay styles.

Based on those streams, *video_generation.py* generates the summary videos for the survey.
Herby, *highlights_state_selection.py* is used to choose one set of states according to the HIGHLIGHTS-DIV algorithm and 10 different random sets of states for the random summaries.
The method that combines those frames to a video is implemented in *image_utils.py*.

# Subfolders
*Action_checks* and *Sanity_checks* check the action distribution of each agent and perform sanity checks for our saliency algorithm.

The videos we used in our survey are stored in the folder *Survey_videos* and the results of this survey are stored and evaluated in *Survey_results*. 
The *models* folder contains the trained agents we used and the streams we used are available upon request.


