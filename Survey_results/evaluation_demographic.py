"""
    This module analyzes the demographic questions about age,gender, Pacman experience and AI experience.
"""

import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from Survey_results.evaluation import show_and_save_plt, analyze_distribution

def analyze_demographic(data):
    """
    Helper function that inserts the AI experience columns into analyze distribution.
    """
    ax = analyze_distribution(data, ['experienceAI2[1]', 'experienceAI2[2]', 'experienceAI2[3]', 'experienceAI2[4]',
                                'experienceAI2[5]'])
    return ax

sns.set(palette= 'colorblind')

data = pd.read_csv('fusion_final.csv')
data['condition'] = data.randnumber.apply(lambda x: 'R' if x==1 else 'H' if x==2 else 'R+S' if x==3 else 'H+S')
data.head()

# AGE 1: <17, 2: 18-24, 3:25-34, 4:35-44, 5:45-54, 6:55-64, 7 >65, 8 not specify
ages = data.age.values
ages = ages[np.where(ages != 8)]
print('age median:', np.median(ages))
print('age mode:', stats.mode(ages))

data_ages = data.loc[data.age != 1]
data['age'] = data.age.apply(lambda x: '<17' if x==1 else '18-24' if x==2 else '25-34' if x==3 else '35-44' if x == 4
                             else '45-54' if x == 5 else '55-64' if x == 6 else '>65' if x == 7 else 'None')
#TODO rewrite show_and_save_plt such that it is not neccessary to do the first plot twice
ax = sns.catplot(x="condition", hue="age", kind="count",
                 data=data,
                 hue_order=['<17', '18-24', '25-34', '35-44', '45-54', '55-64','>65', 'None'],
                order=['R', 'H', 'R+S', 'H+S'], legend=True, palette='colorblind'
                 );
show_and_save_plt(ax, 'demogrpahic/ages', y_label='Number of Participants')
ax = sns.catplot(x="condition", hue="age", kind="count",
                 data=data, aspect= 2,
                 hue_order=['<17', '18-24', '25-34', '35-44', '45-54', '55-64','>65', 'None'],
                order=['R', 'H', 'R+S', 'H+S'], legend=False
                 );
show_and_save_plt(ax, 'demogrpahic/ages', y_label='Number of Participants')


# GENDER 1: male, 2: female, 3:prefer not to answer, 4:other
genders = data.gender.values
genders = genders[np.where(genders != 3)]
genders = genders[np.where(genders != 4)]
genders = genders - 1
print('number females:', genders.sum())
data_gender = data.loc[data.gender < 3]
# set males to 0 and females to 1
data_gender.gender = data_gender.gender.values - 1
ax = sns.barplot(x='condition', y='gender', data=data_gender, order=['R', 'H', 'R+S', 'H+S'])
show_and_save_plt(ax, 'demogrpahic/number_females', y_label='Percentage of Female Participants')

# PACMAN EXP 1: never played, 2: <1year, 3: <5years, 4: >5years ago
pacman_experience = data['experiencePacman'].values
print('pacman exp median:', np.median(pacman_experience))
print('pacman exp mean:', np.mean(pacman_experience))
data['experiencePacman'] = data.experiencePacman.apply(lambda x: 'never' if x == 1 else '< 1 year' if x == 2 else
                                                       '< 5 years' if x == 3 else '> 5 years')
ax = sns.catplot(x="condition", hue="experiencePacman", kind="count", hue_order=['never', '> 5 years', '< 1 year', '< 5 years'],
                 data=data, order=['R', 'H', 'R+S', 'H+S'], legend=False,  aspect= 2);
show_and_save_plt(ax, 'demogrpahic/pacman_experience', y_label='Number of Participants')

# AI VALUES
attitude = data['outcomeAI[1]'].values
attitude = attitude[np.where(attitude != 6)]
print('mean attitude towards AI', np.mean(attitude))

data_attiude = data.loc[data['outcomeAI[1]'] != 6]
ax = sns.barplot(x='condition', y='outcomeAI[1]', data=data_attiude, order=['R', 'H', 'R+S', 'H+S'])
show_and_save_plt(ax, 'demogrpahic/Attitude_towards_AI', y_label='Attitude towards AI')

AI_exp = data.experienceAI.values
print('number of people with Ai experience:', np.sum(AI_exp))

data_random = data.loc[data.randnumber==1]
print('##########Condition 1##########')
ax = analyze_demographic(data_random)
show_and_save_plt(ax, 'demogrpahic/AiExperience_random', y_label='Percent of Participants', ylim=[0,1])

data_highlights = data.loc[data.randnumber==2]
print('##########Condition 2##########')
ax = analyze_demographic(data_highlights)
show_and_save_plt(ax, 'demogrpahic/AiExperience_highlights', y_label='Percent of Participants', ylim=[0,1], label_size = 18, tick_size = 14)

data_randomLRP = data.loc[data.randnumber==3]
print('##########Condition 3##########')
ax = analyze_demographic(data_randomLRP)
show_and_save_plt(ax, 'demogrpahic/AiExperience_randomLRP', y_label='Percent of Participants', ylim=[0,1],label_size = 18, tick_size = 14)

data_highlightsLRP = data.loc[data.randnumber==4]
print('##########Condition 4##########')
ax = analyze_demographic(data_highlightsLRP)
show_and_save_plt(ax, 'demogrpahic/AiExperience_highlightsLRP', y_label='Percent of Participants', ylim=[0,1],label_size = 18, tick_size = 14)

data = data.loc[data.experienceAI == 1]
media = data['experienceAI2[1]'].values
print('I know AI from the media:', np.sum(media))
technology = data['experienceAI2[2]'].values
print("technology:", np.sum(technology))
technology_work = data['experienceAI2[3]'].values
print('technology work:', np.sum(technology_work))
course = data['experienceAI2[4]'].values
print('AI related course', np.sum(course))
research = data['experienceAI2[5]'].values
print('research on AI', np.sum(research))
