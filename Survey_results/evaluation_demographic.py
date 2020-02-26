import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def show_and_save_plt(ax ,file_name, y_label=None, title = None,ylim =None):
    if y_label != None:
        plt.ylabel(y_label)
    #plt.supitle=(title)t
    ax.set(title=title)
    if ylim != None:
        ax.set(ylim=ylim)

    file_name = os.path.join('figures', file_name)
    if not (os.path.isdir(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    plt.savefig(file_name)

    plt.show()

def analyze_demographic(data):

    ax = analyze_distribution(data, ['experienceAI2[1]', 'experienceAI2[2]', 'experienceAI2[3]', 'experienceAI2[4]',
                                'experienceAI2[5]'])
    return ax

def analyze_distribution(data, columns):
    df = data[columns]
    def get_column_number(column):
        if column in columns:
            return columns.index(column) + 1
        else:
            return column

    df = pd.melt(df)
    df.variable =df.variable.apply(get_column_number)
    ax = sns.barplot(x="variable", y="value", data=df)

    return ax

#####  .... questions

data = pd.read_csv('fusion_final.csv')
data['condition'] = data.randnumber.apply(lambda x: 'R' if x==1 else 'H' if x==2 else 'R+S' if x==3 else 'H+S')
data.head()

# 1: <17, 2: 18-24, 3:25-34, 4:35-44, 5:45-54, 6:55-64, 7 >65, 8 not specify
ages = data.age.values
ages = ages[np.where(ages != 8)]
print('age median:', np.median(ages))

data_ages = data.loc[data.age != 1]
ax = sns.boxplot(x='condition', y='age', data=data_ages, order=['R', 'H', 'R+S', 'H+S'])
show_and_save_plt(ax, 'demogrpahic/ages', y_label='age group', ylim=(1,7))


# 1: male, 2: female, 3:prefer not to answer
genders = data.gender.values
genders = genders[np.where(genders != 3)]
genders = genders[np.where(genders != 4)]
genders = genders - 1
print('number females:', genders.sum())

data_gender = data.loc[data.gender < 3]
#get males to 0 and females to 1
data_gender.gender = data_gender.gender.values - 1
ax = sns.barplot(x='condition', y='gender', data=data_gender, order=['R', 'H', 'R+S', 'H+S'])
show_and_save_plt(ax, 'demogrpahic/number_females', y_label='percentage of female participants')

# 1: never played, 2: <1year, 3: <5years, 4: >5years ago
#reorder such that it goes up with experience
data['experiencePacman'] = data.experiencePacman.apply(lambda x: 2 if x == 4 else 4 if x == 2 else x)
pacman_experience = data['experiencePacman'].values
print('pacman exp median:', np.median(pacman_experience))
print('pacman exp mean:', np.mean(pacman_experience))
ax = sns.boxplot(x='condition', y='experiencePacman', data=data, order=['R', 'H', 'R+S', 'H+S'])
show_and_save_plt(ax, 'demogrpahic/pacman_experience', y_label='Pacman experience')


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
show_and_save_plt(ax, 'demogrpahic/AiExperience_random', y_label='percent of participants', ylim=[0,1])

data_highlights = data.loc[data.randnumber==2]
print('##########Condition 2##########')
analyze_demographic(data_highlights)
show_and_save_plt(ax, 'demogrpahic/AiExperience_highlights', y_label='percent of participants', ylim=[0,1])

data_randomLRP = data.loc[data.randnumber==3]
print('##########Condition 3##########')
analyze_demographic(data_randomLRP)
show_and_save_plt(ax, 'demogrpahic/AiExperience_randomLRP', y_label='percent of participants', ylim=[0,1])

data_highlightsLRP = data.loc[data.randnumber==4]
print('##########Condition 4##########')
analyze_demographic(data_highlightsLRP)
show_and_save_plt(ax, 'demogrpahic/AiExperience_highlightsLRP', y_label='percent of participants', ylim=[0,1])


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
