import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def rank_biserial_effect_size(x,y):
    mann_whitney_res = stats.mannwhitneyu(x, y)
    u = mann_whitney_res[0]
    effect_size = 1.0-((2.0*u)/(len(x)*len(y)))
    return effect_size

def mann_whitney(data, column_name):
    ''' doing the mann whitney test for the dependant variable in the given column
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

    print('random < randomLRP')
    print(stats.mannwhitneyu(random_values, randomLRP_values, alternative='less'))
    print(rank_biserial_effect_size(random_values, randomLRP_values))
    print('highlights < highlightsLRP')
    print(stats.mannwhitneyu(highlights_values, highlightsLRP_values, alternative='less'))
    print(rank_biserial_effect_size(highlights_values, highlightsLRP_values))
    print('random < highlights')
    print(stats.mannwhitneyu(random_values, highlights_values, alternative='less'))
    print(rank_biserial_effect_size(random_values, highlights_values))
    print('randomLRP < highlightsLRP')
    print(stats.mannwhitneyu(randomLRP_values, highlightsLRP_values, alternative='less'))
    print(rank_biserial_effect_size(randomLRP_values, highlightsLRP_values))

def show_and_save_plt(ax ,file_name, y_label=None, title = None,ylim =None):
    if y_label != None:
        plt.ylabel(y_label)
    #plt.supitle=(title)t
    #ax.set(title=title)
    if ylim != None:
        ax.set(ylim=ylim)

    # ax.set_ylabel('Entropy (solvers)', fontsize=14)
    # ax.set_xlabel('Condition', fontsize=14)
    ax.yaxis.label.set_size(14)
    ax.xaxis.label.set_size(14)

    file_name = os.path.join('figures', file_name)
    if not (os.path.isdir(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    plt.savefig(file_name)

    plt.show()

def satisfaction_analysis(number, data):
    '''

    :param number: 1 = Retro, 2= trust
    :return:
    '''
    column_name = 'explSatisfaction' + str(number)
    if number == 1:
        result_name =  'explSatisfaction' + 'Retro' + 'Avg'
        title = 'Average user satisfaction for the ' + 'retrospection task'
    elif number == 2:
        result_name = 'explSatisfaction' + 'Trust' + 'Avg'
        title = 'Average user satisfaction for the ' + 'comparison task'
    else:
        print('number not implemented')
    data[column_name + '[3]'] = 6 - data[column_name + '[3]']
    avg_satisfaction =[]
    for i in data.index.values:
        temp = 0
        if data['randnumber'][i] < 3:
            for j in (1,2,3,4):
                temp += data[column_name + '[' + str(j) + ']'][i]
            temp = temp / 4
        else:
            for j in (1, 2, 3, 5, 6):
                temp += data[column_name + '[' + str(j) + ']'][i]
            temp = temp / 5
        avg_satisfaction.append(temp)

    data[result_name] = avg_satisfaction
    # +data['explSatisfaction1[3]_rev']+data['explSatisfaction1[4]']+data['explSatisfaction1[5]']+data['explSatisfaction1[6]']
    ax = sns.barplot(x='condition', y=result_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, result_name, y_label='Average user rating', title=title, ylim=(1, 7))

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

#####Analysis of trust task #####
data = pd.read_csv('fusion_final.csv')

data['TrustClicksTotal'] = data.videoClicks1B + data.videoClicks1A + data.videoClicks2A + data.videoClicks2B + \
                           data.videoClicks3A + data.videoClicks3B

total_number = len(data['TrustClicksTotal'])
#check if they watched enough videos
data = data.loc[data.TrustClicksTotal > 5]
new_number = len(data['TrustClicksTotal'])
difference = total_number - new_number
print('did not watch:', difference)

data['numTrustCorrect'] = data.trust1correct + data.trust2correct + data.trust3correct
data['percentTrustCorrect'] = data.numTrustCorrect.apply(lambda x: x/3.0)
data['condition'] = data.randnumber.apply(lambda x: 'R' if x==1 else 'H' if x==2 else 'R+S' if x==3 else 'H+S')
data.head()


for i in range(1,4):
    column_name = 'trust'+ str(i) +'correct'
    title = 'comparison task for agent ' + str(i)
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), 'percentage of correct selections', title=title)

    column_name = 'trustConfidence' + str(i) + '[confidence1]'
    ax = sns.boxplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(1, 7))

    column_name = 'videoClicks' + str(i)
    data[column_name] = data[column_name+'A'] + data[column_name+'B']
    title = 'pauses during the comparison task for agent ' + str(i)
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents', column_name), 'number of pauses', title=title, ylim=(0, 10))


ax = sns.barplot(x='condition', y='percentTrustCorrect', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax,'TrustPercentCorrect','percentage of correct selections', 'combined trust tasks')

satisfaction_analysis(2, data)


column_name = 'TrustConfAvg'
data[column_name] = (data['trustConfidence2[confidence1]']+data['trustConfidence3[confidence1]']+data['trustConfidence1[confidence1]'])/3.0
ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
show_and_save_plt(ax, column_name, y_label='average confidence',
                  title='Average confidence in the agent selection',ylim=(1,7))

data['trustTimeAvg'] = (data['trust1Time']+data['trust2Time']+data['trust3Time'])/3.0
ax = sns.boxplot(x='condition', y='trustTimeAvg', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax, 'trustTimeAvg', ylim=(0, 500))

data['TrustClicksAvg'] = data['TrustClicksTotal'] / 3
ax = sns.boxplot(x='condition', y='TrustClicksAvg', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax, 'TrustClicksAvg', y_label='average number of clicks', title= 'Pauses of the Video in the trust task', ylim=(0, 10))


mann_whitney(data, 'percentTrustCorrect')
mann_whitney(data, 'explSatisfaction' + 'Trust' + 'Avg')






############ Analysis of Retrospection task ########
data = pd.read_csv('fusion_final.csv')

data['retroClicksTotal'] = data.retrospectionClicks1 + data.retrospectionClicks2 + data.retrospectionClicks3

total_number = len(data['retroClicksTotal'])
#check if the watched enough videos
data = data.loc[data['retroClicksTotal'] > 2]
new_number = len(data['retroClicksTotal'])
difference = total_number - new_number
print('did not watch Retrospection:', difference)

data['condition'] = data.randnumber.apply(lambda x: 'R' if x==1 else 'H' if x==2 else 'R+S' if x==3 else 'H+S')
data.head()




data['retroScoreTotal'] = (data['retrospection1Points']+ data['retrospection2Points']+ data['retrospection3Points'])
data['retroGoalTotal'] = (data['retrospection1Goal']+ data['retrospection2Goal']+ data['retrospection3Goal'])
data['retroPacmanTotal'] = (data['retrospection3[1]'] + data['retrospection3[1]'] + data['retrospection3[1]'])
data['retroClicksTotal'] = data['retroClicksTotal'] / 3

for i in range(1,4):
    column_name = 'retrospection' + str(i) + 'Points'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 3))

    column_name = 'retrospection' + str(i) + 'Goal'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 3))

    column_name = 'retrospection' + str(i) + '[1]'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents','retrospectionPacman'+str(i)), ylim=(0, 3))

    column_name = 'retrospectionClicks' + str(i)
    ax = sns.boxplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 10))



ax = sns.barplot(x='condition', y='retroScoreTotal', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax,'retroScoreTotal',ylim=(0,3))

ax = sns.barplot(x='condition', y='retroGoalTotal', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax,'retroGoalTotal',ylim=(0,3))

ax = sns.barplot(x='condition', y='retroPacmanTotal', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax,'retroPacmanTotal',ylim=(0,3))

ax = sns.boxplot(x='condition', y='retroClicksTotal', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax,'retroClicksAvg',y_label='average number of clicks', title='Pauses of the Video in the analysis task', ylim=(0, 10))

avg_conf = None
for i in range(1,4):
    column_name = 'RetrospectionConf' + str(i) + '[predictConf1]'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), y_label='confidence',ylim=(1,7),
                      title='Confidence in the analysis of agent'+ str(i))

    if i == 1:
        avg_conf = data[column_name]
    else:
        avg_conf += data[column_name]


column_name = 'retroConfAvg'
data[column_name] = avg_conf / 3
ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
show_and_save_plt(ax, column_name, y_label='average confidence',
                  title='Average confidence in the analysis',ylim=(1,7))

data['retroTimeAvg'] = (data['retro1Time']+data['retro2Time']+data['retro3Time'])/3.0
ax = sns.boxplot(x='condition', y='retroTimeAvg', data=data, order=['R','H','R+S','H+S'])
show_and_save_plt(ax, 'retroTimeAvg', ylim=(0, 500))

satisfaction_analysis(1,data)

mann_whitney(data, 'retroScoreTotal')
mann_whitney(data, 'explSatisfaction' + 'Retro' + 'Avg')
