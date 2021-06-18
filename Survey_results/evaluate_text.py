"""
    This module evaluates the textual answers of the participants.
    It also defines how we grouped together the concepts, that were identified by the coder, and
    how those groups should influence the participants score for each agent.
"""

import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from Survey_results.evaluation import show_and_save_plt, rank_biserial_effect_size, mean_and_CI
import os


def mann_whitney(data, column_name):
    ''' doing the mann whitney test for the dependant variable in the given column
        Same es the one in evaluation but we only test H vs HS
    '''
    data_highlights = data.loc[data.randnumber==2]
    data_highlightsLRP = data.loc[data.randnumber==4]

    highlights_values = data_highlights[column_name].values
    highlightsLRP_values = data_highlightsLRP[column_name].values

    print('##### testing the column:', column_name,'#####')

    print('CI highlights:')
    mean_and_CI(highlights_values)
    print('CI highlights LRP:')
    mean_and_CI(highlightsLRP_values)

    p_vals = []
    print('highlights < highlightsLRP')
    print(stats.mannwhitneyu(highlights_values, highlightsLRP_values, alternative='less'))
    print(rank_biserial_effect_size(highlights_values, highlightsLRP_values))
    p_vals.append(stats.mannwhitneyu(highlights_values, highlightsLRP_values, alternative='less')[1])


def mean_analysis(data, column_name):
    ''' calculating mean and CI for the dependant variable in the given column
    '''
    data_random = data.loc[data.randnumber==1]
    data_randomLRP = data.loc[data.randnumber==3]
    data_highlights = data.loc[data.randnumber==2]
    data_highlightsLRP = data.loc[data.randnumber==4]

    random_values = data_random[column_name].values
    randomLRP_values = data_randomLRP[column_name].values
    highlights_values = data_highlights[column_name].values
    highlightsLRP_values = data_highlightsLRP[column_name].values

    print('##### testing the column:', column_name,'#####')

    print('CI random:')
    mean_and_CI(random_values)
    print('CI random_LRP:')
    mean_and_CI(randomLRP_values)
    print('CI highlights:')
    mean_and_CI(highlights_values)
    print('CI highlights LRP:')
    mean_and_CI(highlightsLRP_values)


def read_text_data(file_name):
    '''
    reads the data frame that contains the categorized textuals answers
    :param file_name: csv file, expects sepration by ';'
    :return: the data frame and an array with the participants' seeds
    '''
    text_data = pd.read_csv(file_name, sep=';')
    concepts = text_data['SESSION ID: SEED'].values

    return text_data, concepts


def get_grouped_df(categories_df, data,groups):
    '''
    generates a data frame that encodes for each group if the participants answer contained a category that belongs to
    this group
    :param categories_df: csv file with the categories identified by the coder, ecpects seperation by ';'
    :param data: the survey results as exported from LimeSurvey
    :param groups: a dictionary that defnies how the categories should be grouped
    :return: the resulting data frame, the first column contains the participants seed, the other columns encode
     , for each group, if it was part of the participants answer
    '''
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


def count_non_grouped_items(data, categories_df, groups):
    '''
    helper function to check how many categories have not been grouped yet
    (counts the actual appearances in the participants answers)
    '''
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


def score(positive, neutral, data):
    '''
    a simple scoring function that gives +1 for each correct group and -1 for false group, neutral groups are ignored
    :param positive: array of positive groups
    :param neutral: array of neutral groups
    :param data: data frame encoding for each participant and group, if the group is contained in the participants answers
    :return: array of scores for each participant
    '''
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


#dictionary of the grouped categories:
groups = {'eat_power_pill': ['eating PP', 'eating as many PP as possible', 'eat PP when ghosts are near',
                             'prioritizing PP', 'prioritizing PP to eat ghosts', 'prioritizing PP , but not eat ghosts',
                             'eat PP to get points'],

          'ignore_power_pill': ['do not care about PP'],

          'eat_normal_pill': ['eat NP to get points', 'eating NP','eating as many NP as possible', 'prioritizing NP',
                              'clearing the stage'],

          'ignore_normal_pill': ['do not care about NP', 'focus on areas wihtout NP'],

          'avoid_ghost': ['avoiding G', 'avoiding G strongly', 'wait for G to go away', 'outmanoveuring G',
                          'hiding from G', 'mislead ghosts','avoids being eaten / caught',
                          'avoiding to lose / staying alive', 'stays away from danger'],

          'move_towards_ghost' : ['being close to G', 'trying to eat G NON blue', '(easily) caught by G',
                                  'easily caught by G'],

          'ignore_ghosts': ['do not care about G'],

          'making_ghosts_blue': ['making G blue'],

          'eat_blue_ghost': ['being close to blue G','eating as many G as possible','eat blue G to get points',
                             'chasing/going for G','eating the blue G', 'eating to jail many G',
                             'prioritizing PP to eat ghosts'],
          'avoid_blue_ghost': ['avoiding blue G'],

          'ignore_blue_ghost': ['do not care about blue G','prioritizing PP , but not eat ghosts'],

          'eat_cherry': ['prioritizing cherry', 'eat cherry to get points', 'going for cherry', 'eating cherry'],

          'ignore_cherry': ['do not care about cherry'],

          'random_movement': ['moving randomly', 'move all over map','switching directions /back&forth',
                              'not moving / being stuck', 'sticking to walls / outside', 'confused',
                              'without strategy /random', 'not planning ahead', 'switching directions'],

          'focus_on_Pacman': ['focus on PM', 'focus on whats in front of/around PM', 'stuck to itself'],

          'staying_in_corners': ['staying in corners'],

          'RULES': ['RULES'],
          'HEATMAP': ['HEATMAP'],
          'INTERPRETATION': ['INTERPRETATION'],
          'GAMEPLAY': ['GAMEPLAY'],
          'UNJUSTIFIED': ['UNJUSTIFIED', 'UNRELATED'],
          #'UNRELATED' : ['UNRELATED']
          'UNDECIDED': ['SAME SAME']
          }

# rating of the category groups defined above, stored as arrays
positive_agent1 = ['eat_power_pill','ignore_normal_pill','ignore_ghosts','ignore_blue_ghost','ignore_cherry','focus_on_Pacman', 'staying_in_corners' ]
neutral_agent1 = ['eat_normal_pill','making_ghosts_blue']
main_goal_agent1 = 'eat_power_pill'

positive_agent2 = ['ignore_cherry','focus_on_Pacman','making_ghosts_blue','eat_blue_ghost']
neutral_agent2 = ['eat_normal_pill', 'eat_power_pill', 'ignore_ghosts']
main_goal_agent2 = 'eat_blue_ghost'

positive_agent3 = ['ignore_cherry','focus_on_Pacman','making_ghosts_blue','eat_blue_ghost','avoid_ghost']
neutral_agent3 = ['eat_normal_pill', 'eat_power_pill']
main_goal_agent3 = 'avoid_ghost'

# the variables to be further analyzed
interesting =['focus_on_Pacman', 'score', 'goal', 'UNJUSTIFIED', 'GAMEPLAY', 'HEATMAP']

if __name__ == '__main__':
    sns.set(palette='colorblind')
    sns.set_style("whitegrid")

    data = pd.read_csv('fusion_final.csv')

    #### RETROSPECTION TASK ####

    #skip users who did not watch retro:
    data['retroClicksTotal'] = data.retrospectionClicks1 + data.retrospectionClicks2 + data.retrospectionClicks3

    total_number = len(data['retroClicksTotal'])
    # check if the participant watched enough videos
    data = data.loc[data['retroClicksTotal'] > 2]
    new_number = len(data['retroClicksTotal'])
    difference = total_number - new_number
    print('did not watch Retrospection:', difference)

    data['condition'] = data.randnumber.apply(
        lambda x: 'L' if x == 1 else 'H' if x == 2 else 'L+S' if x == 3 else 'H+S')

    #### SINGLE AGENTS ####

    ### agent 1 ###
    new_data_frame = get_grouped_df('PacmanStrategies_INT1_EXP1.CSV',data, groups)
    new_data_frame['score'] = score(positive_agent1, neutral_agent1, new_data_frame)
    new_data_frame['goal'] = new_data_frame[main_goal_agent1]

    new_data_frame.to_csv('text_intention1.csv')

    merge1 = pd.merge(data,new_data_frame,on='seed')

    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge1, order=['L', 'H', 'L+S', 'H+S'])
        plt.show()

    ### agent 2 ###
    new_data_frame = get_grouped_df('PacmanStrategies_INT2_EXP2.CSV', data, groups)
    new_data_frame['score'] = score(positive_agent2, neutral_agent2, new_data_frame)
    new_data_frame['goal'] = new_data_frame[main_goal_agent2]

    new_data_frame.to_csv('text_intention2.csv')

    merge2 = pd.merge(data, new_data_frame, on='seed')

    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge2, order=['L', 'H', 'L+S', 'H+S'])
        plt.show()

    ### agent 3 ###
    new_data_frame = get_grouped_df('PacmanStrategies_INT3_EXP3.CSV', data, groups)
    new_data_frame['score'] = score(positive_agent3, neutral_agent3, new_data_frame)
    new_data_frame['goal'] = new_data_frame[main_goal_agent3]

    new_data_frame.to_csv('text_intention3.csv')

    merge3 = pd.merge(data, new_data_frame, on='seed')

    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge3, order=['L', 'H', 'L+S', 'H+S'])
        plt.show()

    ax = sns.barplot(x='condition', y='avoid_ghost', data=merge3, order=['L', 'H', 'L+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('text', 'avoid_ghost_agent3'), y_label= 'got avoid ghost')

    #### COMBINED AGENTS ####

    #summing and plotting
    for key in interesting:
        factor = 1
        ylim = (0,3)
        if key == 'focus_on_Pacman':
            label = 'Number of Mentions'
            factor = 1
        elif key == 'goal':
            label = 'Number of Mentions'
        elif key == 'score':
            label = 'Total Score'
            ylim = None
        elif key in ['HEATMAP', 'UNJUSTIFIED', 'GAMEPLAY']:
            label = 'Number of Mentions'
        else:
            label = 'average ' + key
        merge1[key] += merge2[key] + merge3[key]
        merge1[key] = merge1[key] / factor
        ax = sns.barplot(x='condition', y=key, data=merge1, order=['L', 'H', 'L+S', 'H+S'])
        if key == 'HEATMAP':
            ax = sns.barplot(x='condition', y=key, data=merge1, order=['L+S', 'H+S'],
                             palette=[sns.color_palette()[2], sns.color_palette()[3]])
        show_and_save_plt(ax, os.path.join('text', key + '_total'), y_label= label, ylim= ylim)

    #stats for the total values
    mann_whitney(merge1,'score')
    mean_analysis(merge1, 'HEATMAP')
    mean_analysis(merge1, 'UNJUSTIFIED')
    mean_analysis(merge1, 'GAMEPLAY')

    mean_analysis(merge1, 'goal')
    mean_analysis(merge1, 'focus_on_Pacman')


    #### TRUST TASK ####

    #undecided exists in the trust task as additional category
    interesting =['UNJUSTIFIED', 'GAMEPLAY', 'HEATMAP','UNDECIDED']

    data = pd.read_csv('fusion_final.csv')

    # skip users who did not watch retro:
    data['TrustClicksTotal'] = data.videoClicks1B + data.videoClicks1A + data.videoClicks2A + data.videoClicks2B + \
                               data.videoClicks3A + data.videoClicks3B

    total_number = len(data['TrustClicksTotal'])
    # check if they watched enough videos
    data = data.loc[data.TrustClicksTotal > 5]
    new_number = len(data['TrustClicksTotal'])
    difference = total_number - new_number
    print('did not watch:', difference)

    data['condition'] = data.randnumber.apply(
        lambda x: 'L' if x == 1 else 'H' if x == 2 else 'L+S' if x == 3 else 'H+S')

    #### SINGLE AGENTS ####

    ### comparison 1 ###
    new_data_frame = get_grouped_df('PacmanStrategies_TRUST_1.CSV', data, groups)
    new_data_frame.to_csv('text_trust1.csv')
    merge1 = pd.merge(data, new_data_frame, on='seed')
    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge1, order=['L', 'H', 'L+S', 'H+S'])
        plt.show()

    ### comparison 2 ###
    new_data_frame = get_grouped_df('PacmanStrategies_TRUST_2.CSV', data, groups)
    new_data_frame.to_csv('text_trust2.csv')
    merge2 = pd.merge(data, new_data_frame, on='seed')
    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge2, order=['L', 'H', 'L+S', 'H+S'])
        plt.show()

    ### comparison 3 ###
    new_data_frame = get_grouped_df('PacmanStrategies_TRUST_3.CSV', data, groups)
    new_data_frame.to_csv('text_trust3.csv')
    merge3 = pd.merge(data, new_data_frame, on='seed')
    for key in interesting:
        ax = sns.barplot(x='condition', y=key, data=merge3, order=['L', 'H', 'L+S', 'H+S'])
        plt.show()

    #### COMBINED AGENTS ####

    # summing and plotting
    for key in interesting:
        merge1[key] += merge2[key] + merge3[key]
        merge1[key] = merge1[key]
        ax = sns.barplot(x='condition', y=key, data=merge1, order=['L', 'H', 'L+S', 'H+S'])
        if key == 'HEATMAP':
            ax = sns.barplot(x='condition', y=key, data=merge1, order=['L+S', 'H+S'], palette=[sns.color_palette()[2],sns.color_palette()[3]])
        if key in ['HEATMAP', 'UNJUSTIFIED', 'GAMEPLAY']:
            label = 'Number of Mentions'
        else:
            label = 'average ' + key
        show_and_save_plt(ax, os.path.join('text', 'trust_' + key + '_total'), y_label=label, ylim=(0,3))

    #stats for the total values
    mean_analysis(merge1, 'HEATMAP')
    mean_analysis(merge1, 'UNJUSTIFIED')
    mean_analysis(merge1, 'GAMEPLAY')

    pass