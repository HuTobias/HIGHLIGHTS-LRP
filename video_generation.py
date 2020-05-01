"""
    Utility script to generate all summarys for a given stream at once (1 HIGHLIGHTS-DIV summary and 10 random summaries).
"""


import pandas as pd
import image_utils
import numpy as np
from highlights_state_selection import read_q_value_files, read_feature_files, compute_states_importance, highlights_div, random_state_selection, read_input_files
import os

#random seeds for the random summaries
seeds=[ 42, 1337, 1, 7, 13, 21, 153, 90,19234761, 291857957]

if __name__ == '__main__':

    def help_function(stream_folder,features='input', load_states=False):
        trajectories = 5
        context = 10
        minimum_gap = 10
        parameter_string = str(trajectories) + '_' + str(context) + '_' + str(minimum_gap)
        video_folder = os.path.join(stream_folder,'smooth_stream_vid_max/')

        q_values_df = read_q_value_files(stream_folder + '/q_values')
        states_q_values_df = compute_states_importance(q_values_df, compare_to='second')
        states_q_values_df.to_csv(stream_folder + '/states_importance_second.csv')
        states_q_values_df = pd.read_csv(stream_folder + '/states_importance_second.csv')
        if features == 'features':
            features_df = read_feature_files(stream_folder + '/features')
            features_df.to_csv(stream_folder + '/state_features.csv')
            features_df = pd.read_csv(stream_folder + '/state_features.csv')
            state_features_importance_df = pd.merge(states_q_values_df, features_df, on='state')
            state_features_importance_df = state_features_importance_df[['state', 'q_values', 'importance', 'features']]
            state_features_importance_df.to_csv(stream_folder + '/state_features_impoartance.csv')
            state_features_importance_df = pd.read_csv(stream_folder + '/state_features_impoartance.csv')

            state_features_importance_df['features'] = state_features_importance_df['features'].apply(lambda x:
                                                                                                      np.fromstring(
                                                                                                          x.replace('\n', '')
                                                                                                              .replace('[', '')
                                                                                                              .replace(']', '')
                                                                                                              .replace('  ',
                                                                                                                       ' '),
                                                                                                          sep=' '))
        elif features == 'input': #we need an extra case since the input arrays are too big to be saved in csv.
            features_df = read_input_files(stream_folder + '/state')
            state_features_importance_df = pd.merge(states_q_values_df, features_df, on='state')
            state_features_importance_df = state_features_importance_df[['state', 'q_values', 'importance', 'features']]
        else:
            print('feature type not support.')

        if load_states:
            summary_states_with_context = np.load(stream_folder + '/summary_states_with_context.npy')
        else:
            summary_states, summary_states_with_context = highlights_div(state_features_importance_df, trajectories,
                                                                     context, minimum_gap)
            with open(stream_folder + '/summary_states.txt', "w") as text_file:
                text_file.write(str(summary_states))
            np.save(stream_folder + '/summary_states_with_context.npy', summary_states_with_context)

        image_folder = os.path.join(stream_folder, 'argmax_smooth/')
        video_name = 'highlights_div_lrp_' + parameter_string + '.mp4'
        image_utils.generate_video(image_folder, video_folder, video_name,
                                   image_indices=summary_states_with_context)

        image_folder = os.path.join(stream_folder, 'screen_smooth/')
        video_name = 'highlights_div_' + parameter_string + '.mp4'
        image_utils.generate_video(image_folder, video_folder, video_name,
                                   image_indices=summary_states_with_context)

        for i in range(10):
            seed = seeds[i]
            random_states, random_states_with_context = random_state_selection(state_features_importance_df,
                                                                               trajectories,
                                                                               context, minimum_gap,seed=seed)

            image_folder = os.path.join(stream_folder, 'argmax_smooth/')
            video_name = 'random_lrp_' + str(i+1) + '_' + parameter_string + '.mp4'
            image_utils.generate_video(image_folder, video_folder, video_name,
                                       image_indices=random_states_with_context)

            image_folder = os.path.join(stream_folder, 'screen_smooth/')
            video_name = 'random_' + str(i+1) + '_' + parameter_string + '.mp4'
            image_utils.generate_video(image_folder, video_folder, video_name,
                                       image_indices=random_states_with_context)

    stream_folder = 'stream'
    help_function(stream_folder, load_states=False)
