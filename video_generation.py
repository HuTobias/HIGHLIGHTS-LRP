import pandas as pd
import image_utils
import numpy as np
from highlights_state_selection import highlights, highlights_div, random_state_selection
import os

#utility script to generate all summarys at once

if __name__ == '__main__':

    def help_function(stream_folder):
        video_folder = os.path.join(stream_folder,'smooth_vid/')
        states_importance = 'state_features_importance.csv'

        state_features_importance_df = pd.read_csv(states_importance)
        state_features_importance_df['features'] = state_features_importance_df['features'].apply(lambda x:
                                                                                                  np.fromstring(
                                                                                                      x.replace('\n', '')
                                                                                                          .replace('[', '')
                                                                                                          .replace(']', '')
                                                                                                          .replace('  ',
                                                                                                                   ' '),
                                                                                                      sep=' '))

        summary_states, summary_states_with_context = highlights_div(state_features_importance_df, 15, 10, 10)

        random_states, random_states_with_context = random_state_selection(state_features_importance_df, 15, 10, 10)

        image_folder = os.path.join(stream_folder, 'test/argmax_smooth/')
        video_name = 'highlights_div_15_10_10_lrp.mp4'
        image_utils.generate_video(image_folder, video_folder, video_name,
                                   image_indices=summary_states_with_context)
        image_utils.generate_video(image_folder, video_folder, 'random_15_10_10_lrp.mp4',
                                   image_indices=random_states_with_context)

        image_folder = os.path.join(stream_folder, 'screen/')
        video_name = 'highlights_div_15_10_10.mp4'
        image_utils.generate_video(image_folder, video_folder, video_name,
                                   image_indices=summary_states_with_context)
        image_utils.generate_video(image_folder, video_folder, 'random_15_10_10.mp4',
                                   image_indices=random_states_with_context)

        #for testing different channels
        # image_folder = os.path.join(stream_folder, 'test/c0/')
        # video_name = 'highlights_div_15_10_10_lrp_c0.mp4'
        # image_utils.generate_video(image_folder, video_folder, video_name,
        #                            image_indices=summary_states_with_context)
        #
        # image_folder = os.path.join(stream_folder, 'test/c1/')
        # video_name = 'highlights_div_15_10_10_lrp_c1.mp4'
        # image_utils.generate_video(image_folder, video_folder, video_name,
        #                            image_indices=summary_states_with_context)

    help_function('stream_2M')

    help_function('stream_1M')

    help_function('stream_500k')
