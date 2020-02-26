import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from Survey_results.evaluation import show_and_save_plt, analyze_distribution
import os

def read_text_data(file_name):
    text_data = pd.read_csv(file_name, sep=';')
    concepts = text_data['SESSION ID: SEED'].values

    return text_data, concepts

def get_grouped_df(categories_df, data):
    text_data, concepts = read_text_data(categories_df)

    seeds = data['seed']
    randnumbers = data['randnumber']
    new_randnumbers = []
    new_seeds = []

    # create empty lists to fill the dataframe later
    new_groups = {}
    for key in groups.keys():
        new_groups[key] = []

    for idx in data.index.values:
        # getting the condition number
        randnumber = randnumbers[idx]
        new_randnumbers.append(randnumber)

        # unique seed of the participant
        seed = seeds[idx]
        new_seeds.append(seed)

        # get indices of concepts included in the participants answer
        try:
            test = text_data[str(seed)].values
        except:
            print('seed', seed, 'not found')
            pass
        categories = np.where(test == 1)
        user_list = concepts[categories]
        # print(user_list)
        for key in groups.keys():
            temp = check_group(user_list, groups[key])
            new_groups[key].append(temp)

    new_data_frame = pd.DataFrame()
    # new_data_frame['randnumber'] = new_randnumbers
    new_data_frame['seed'] = new_seeds
    for key in groups.keys():
        new_data_frame[key] = new_groups[key]

    return new_data_frame

def count_non_grouped_items(data, categories_df):
    text_data, concepts = read_text_data(categories_df)

    seeds = data['seed']
    randnumbers = data['randnumber']
    new_randnumbers = []
    new_seeds = []

    non_category = 0
    for idx in data.index.values:
        # getting the condition number
        randnumber = randnumbers[idx]
        new_randnumbers.append(randnumber)

        # unique seed of the participant
        seed = seeds[idx]

        # get indices of concepts included in the participants answer
        try:
            test = text_data[str(seed)].values
        except:
            print('seed', seed, 'not found')
            pass
        categories = np.where(test == 1)[0]
        for cat in categories:
            present = False
            for i in groups.values():
                if concepts[cat] in i:
                    present = True
            if present == False:
                print(concepts[cat])
                non_category += 1

    print(non_category)

def check_group(user_list,group_list):
    ''' check if atleast one item from the group list is in the user_list
    '''
    for i in group_list:
        if i in user_list:
            return 1
    return 0

#dictionary of the grouped categories:
groups = {'eat_power_pill': ['eating PP','eating as many PP as possible','eat PP when ghosts are near','prioritizing PP',
                             'prioritizing PP to eat ghosts','prioritizing PP , but not eat ghosts','eat PP to get points','clearing the stage'],
          'ignore_power_pill': ['do not care about PP'],

          'eat_normal_pill' : ['eat NP to get points','eating NP','eating as many NP as possible','prioritizing NP'],
          'ignore_normal_pill' : ['do not care about NP','focus on areas wihtout NP'],

          'avoid_ghost' : ['avoiding G','avoiding G strongly','wait for G to go away','outmanoveuring G','hiding from G',
                           'mislead ghosts','avoids being eaten / caught','avoiding to lose / staying alive','stays away from danger'],
          'move_towards_ghost' : ['being close to G','trying to eat G NON blue','(easily) caught by G', 'easily caught by G'],
          'ignore_ghosts' : ['do not care about G'],

          'making_ghosts_blue':['making G blue'],
          'eat_blue_ghost': ['being close to blue G','eating as many G as possible','eat blue G to get points',
                             'chasing/going for G','eating the blue G', 'eating to jail many G', 'prioritizing PP to eat ghosts'],
          'avoid_blue_ghost': ['avoiding blue G'],
          'ignore_blue_ghost': ['do not care about blue G','prioritizing PP , but not eat ghosts'],

          'eat_cherry' : ['prioritizing cherry', 'eat cherry to get points', 'going for cherry','eating cherry'],
          'ignore_cherry' : ['do not care about cherry'],

          'random_movement' : ['moving randomly','move all over map','switching directions /back&forth',
                               'not moving / being stuck', 'sticking to walls / outside', 'confused',
                               'without strategy /random', 'not planning ahead', 'switching directions'],

          'focus_on_Pacman' : ['focus on PM','focus on whats in front of/around PM', 'stuck to itself'],
          'staying_in_corners' : ['staying in corners'],

          'RULES' : ['RULES'],
          'HEATMAP' : ['HEATMAP'],
          'INTERPRETATION' : ['INTERPRETATION'],
          'GAMEPLAY' : ['GAMEPLAY'],
          'UNJUSTIFIED' : ['UNJUSTIFIED','UNRELATED']
          #'UNRELATED' : ['UNRELATED']
          }
positive_agent1 = ['eat_power_pill','ignore_normal_pill','ignore_ghosts','ignore_blue_ghost','ignore_cherry','focus_on_Pacman', 'staying_in_corners' ]
neutral_agent1 = ['eat_normal_pill','making_ghosts_blue']

positive_agent2 = ['ignore_cherry','focus_on_Pacman','making_ghosts_blue','eat_blue_ghost']
neutral_agent2 = ['eat_normal_pill', 'eat_power_pill', 'ignore_ghosts']

positive_agent3 = ['ignore_cherry','focus_on_Pacman','making_ghosts_blue','eat_blue_ghost','avoid_ghost']
neutral_agent3 = ['eat_normal_pill', 'eat_power_pill']

def score(positive, neutral, data):
    keys = groups.keys() -['RULES', 'HEATMAP', 'INTERPRETATION', 'GAMEPLAY', 'UNJUSTIFIED']
    sum = np.zeros(data.index.values.shape)
    for key in keys:
        if key in positive:
            sum += data[key]
        elif key in neutral:
            pass
        else:
            sum -= data[key]
    return sum



interesting =['focus_on_Pacman', 'UNJUSTIFIED', 'score']

if __name__ == '__main__':

    data = pd.read_csv('fusion_final.csv')

    #skip users who did not watch retro:
    data['retroClicksTotal'] = data.retrospectionClicks1 + data.retrospectionClicks2 + data.retrospectionClicks3

    total_number = len(data['retroClicksTotal'])
    # check if the watched enough videos
    data = data.loc[data['retroClicksTotal'] > 2]
    new_number = len(data['retroClicksTotal'])
    difference = total_number - new_number
    print('did not watch Retrospection:', difference)

    data['condition'] = data.randnumber.apply(
        lambda x: 'R' if x == 1 else 'H' if x == 2 else 'R+S' if x == 3 else 'H+S')

    new_data_frame = get_grouped_df('PacmanStrategies_INT1_EXP1.CSV',data)
    new_data_frame['score'] = score(positive_agent1, neutral_agent1, new_data_frame)

    new_data_frame.to_csv('int1.csv')

    merge1 = pd.merge(data,new_data_frame,on='seed')

    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge1, order=['R','H','R+S','H+S'])
        plt.show()

    new_data_frame = get_grouped_df('PacmanStrategies_INT2_EXP2.CSV', data)
    new_data_frame['score'] = score(positive_agent2, neutral_agent2, new_data_frame)

    new_data_frame.to_csv('int2.csv')

    merge2 = pd.merge(data, new_data_frame, on='seed')

    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge2, order=['R', 'H', 'R+S', 'H+S'])
        plt.show()

    new_data_frame = get_grouped_df('PacmanStrategies_INT3_EXP3.CSV', data)
    new_data_frame['score'] = score(positive_agent3, neutral_agent3, new_data_frame)

    new_data_frame.to_csv('int3.csv')

    merge3 = pd.merge(data, new_data_frame, on='seed')

    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge3, order=['R', 'H', 'R+S', 'H+S'])
        plt.show()

    ax = sns.barplot(x='condition', y='avoid_ghost', data=merge3, order=['R', 'H', 'R+S', 'H+S'])
    plt.show()

    #summing
    for key in interesting:
        merge1[key] += merge2[key] + merge3[key]
        ax = sns.barplot(x='condition', y=key, data=merge1, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('text', key + '_total'), y_label= key + ' total')




    pass