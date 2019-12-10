import numpy as np
import pandas as pd
import os


def highlights(state_importance_df, budget, context_length):
    top_states = state_importance_df.nlargest(budget, 'importance')['state'].values
    return top_states


def compute_states_importance(states_q_values_df):
    states_q_values_df['importance'] = states_q_values_df['q_values'].apply(lambda x: np.max(x)-np.min(x))
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
    test_data = pd.DataFrame({'state':[1,2,3],'q_values':[[1,2,3],[1,1,1],[2,1,1]]})
    # print(highlights(test_data,2))
    q_values_df = read_q_value_files('q_values')
    states_q_values_df = compute_states_importance(q_values_df)
    states_q_values_df.to_csv('states_importance.csv')
