import numpy as np
import pandas as pd
import os
from bisect import bisect
from bisect import insort_left


def highlights(state_importance_df, budget, context_length, minimum_gap):
    ''' generate highlights summary
    :param state_importance_df: dataframe with 2 columns: state and importance score of the state
    :param budget: allowed length of summary - note this includes only the important states, it doesn't count context
    around them
    :param context_length: how many states to show around the chosen important state (e.g., if context_lenght=10, we
    will show 10 states before and 10 states after the important state
    :param minimum_gap: how many states should we skip after showing the context for an important state. For example, if
    we chose state 200, and the context length is 10, we will show states 189-211. If minimum_gap=10, we will not
    consider states 212-222 and states 178-198 because they are too close
    :return: a list with the indices of the important states. Note, it currently doesn't return the indices for the
    context around them, we can add that here or handle it when we create the visual summary
    '''
    sorted_df = state_importance_df.sort_values(['importance'], ascending=False)
    summary_states = []
    for index, row in sorted_df.iterrows():

        state_index = row['state']
        index_in_summary = bisect(summary_states, state_index)
        # print('state: ', state_index)
        # print('index in summary: ', index_in_summary)
        # print('summary: ', summary_states)
        state_before = None
        state_after = None
        if index_in_summary > 0:
            state_before = summary_states[index_in_summary-1]
        if index_in_summary < len(summary_states):
            state_after = summary_states[index_in_summary]
        if state_after is not None:
            if state_index+context_length+minimum_gap > state_after:
                continue
        if state_before is not None:
            if state_index-context_length-minimum_gap < state_before:
                continue
        insort_left(summary_states,state_index)
        if len(summary_states) == budget:
            break

    return summary_states


def compute_states_importance(states_q_values_df, compare_to='worst'):
    if compare_to == 'worst':
        states_q_values_df['importance'] = states_q_values_df['q_values'].apply(lambda x: np.max(x)-np.min(x))
    elif compare_to == 'second':
        states_q_values_df['importance'] = states_q_values_df['q_values'].apply(lambda x: np.max(x)-np.partition(x.flatten(), -2)[-2])
    return states_q_values_df

def read_q_value_files(path):
    ''' reading q values from files. Assume each state is a seperate text file with a list of q values
    :param path: path to the directory where the text files are stored
    :return: a pandas dataframe with two columns: state (index) and q_values (numpy array)
    '''
    states = []
    q_values_list = []
    for filename in os.listdir(path):
        file_split = filename.split('_')
        state_index = int(file_split[len(file_split)-1][:-4])
        states.append(state_index)
        # print(filename)
        with open(path+'/'+filename, 'r') as q_val_file:
            q_vals = str.strip(q_val_file.read(),'[]')
            q_vals = np.fromstring(q_vals,dtype=float, sep=' ')
            q_values_list.append(q_vals)

    q_values_df = pd.DataFrame({'state':states, 'q_values':q_values_list})
    return q_values_df

if __name__ == '__main__':
    # test_data = pd.DataFrame({'state':[1,2,3],'q_values':[[1,2,3],[1,1,1],[2,1,1]]})
    # # print(highlights(test_data,2))
    # q_values_df = read_q_value_files('q_values')
    # states_q_values_df = compute_states_importance(q_values_df, compare_to='second')
    # # print(highlights(highlights,20,10,10))
    # states_q_values_df.to_csv('states_importance_second.csv')
    states_q_values_df = pd.read_csv('states_importance_second.csv')
    print(highlights(states_q_values_df,20,10,10))
    # a = [1,4,6,10]
    # print(bisect(a,7))
