import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import os



def rank_biserial_effect_size(x,y):
    mann_whitney_res = stats.mannwhitneyu(x, y)
    u = mann_whitney_res[0]
    effect_size = 1.0-((2.0*u)/(len(x)*len(y)))
    return effect_size

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
    # TODO forgot the task specific stuff
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
    ax = sns.barplot(x='condition', y=result_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, result_name, y_label='Average user rating', title=title, ylim=(1, 7))



#####Analysis of trust task #####
data = pd.read_csv('fusion_trust_eval.csv')

data['TrustClicksTotal'] = data.videoClicks1B + data.videoClicks1A + data.videoClicks2A + data.videoClicks2B + \
                           data.videoClicks3A + data.videoClicks3B

total_number = len(data['TrustClicksTotal'])
#check if they watched enough videos
data = data.loc[data.TrustClicksTotal > -1]
new_number = len(data['TrustClicksTotal'])
difference = total_number - new_number
print('did not watch:', difference)

data['numTrustCorrect'] = data.trust1correct + data.trust2correct + data.trust3correct
data['percentTrustCorrect'] = data.numTrustCorrect.apply(lambda x: x/3.0)
data['condition'] = data.randnumber.apply(lambda x: 'R' if x==1 else 'HL' if x==2 else 'R-S' if x==3 else 'HL-S')
data.head()


for i in range(1,4):
    column_name = 'trust'+ str(i) +'correct'
    title = 'comparison task for agent ' + str(i)
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), 'percentage of correct selections', title=title)

    column_name = 'trustConfidence' + str(i) + '[confidence1]'
    ax = sns.boxplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(1, 7))

    column_name = 'videoClicks' + str(i)
    data[column_name] = data[column_name+'A'] + data[column_name+'B']
    title = 'pauses during the comparison task for agent ' + str(i)
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents', column_name), 'number of pauses', title=title)


ax = sns.boxplot(x='condition', y='percentTrustCorrect', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax,'percentTrustCorrect','percentage of correct selections', 'combined trust tasks')

satisfaction_analysis(2, data)

data['trustConfidenceAvg'] = (data['trustConfidence2[confidence1]']+data['trustConfidence3[confidence1]']+data['trustConfidence1[confidence1]'])/3.0
ax = sns.boxplot(x='condition', y='trustConfidenceAvg', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax,'trustConfidenceAvg',ylim=(1,7))

data['trustTimeAvg'] = (data['trust1Time']+data['trust2Time']+data['trust3Time'])/3.0
ax = sns.boxplot(x='condition', y='trustTimeAvg', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax, 'trustTimeAvg', ylim=(0, 500))

data['TrustClicksAvg'] = data['TrustClicksTotal'] / 3
ax = sns.boxplot(x='condition', y='TrustClicksAvg', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax, 'TrustClicksAvg', y_label='average number of clicks', title= 'Pauses of the Video in the trust task')







############ Analysis of Retrospection task ########
data = pd.read_csv('fusion_trust_eval.csv')

data['RetroClicksTotal'] = data.retrospectionClicks1 + data.retrospectionClicks2 + data.retrospectionClicks3

total_number = len(data['RetroClicksTotal'])
#check if the watched enough videos
data = data.loc[data.RetroClicksTotal > 2]
new_number = len(data['RetroClicksTotal'])
difference = total_number - new_number
print('did not watch Retrospection:', difference)

data['condition'] = data.randnumber.apply(lambda x: 'R' if x==1 else 'HL' if x==2 else 'R-S' if x==3 else 'HL-S')
data.head()




data['retroScoreTotal'] = (data['retrospection1Points']+ data['retrospection2Points']+ data['retrospection3Points'])
data['retroGoalTotal'] = (data['retrospection1Goal']+ data['retrospection2Goal']+ data['retrospection3Goal'])
data['retroPacmanTotal'] = (data['retrospection3[1]'] + data['retrospection3[1]'] + data['retrospection3[1]'])
data['retroClicksTotal'] = (data['retrospectionClicks1'] + data['retrospectionClicks2'] + data['retrospectionClicks3']) / 3

for i in range(1,4):
    column_name = 'retrospection' + str(i) + 'Points'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 3))

    column_name = 'retrospection' + str(i) + 'Goal'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 3))

    column_name = 'retrospection' + str(i) + '[1]'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents','retrospectionPacman'+str(i)), ylim=(0, 3))

    column_name = 'retrospectionClicks' + str(i)
    ax = sns.boxplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name))



ax = sns.barplot(x='condition', y='retroScoreTotal', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax,'retroScoreTotal',ylim=(0,3))

ax = sns.barplot(x='condition', y='retroGoalTotal', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax,'retroGoalTotal',ylim=(0,3))

ax = sns.barplot(x='condition', y='retroPacmanTotal', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax,'retroPacmanTotal',ylim=(0,3))

ax = sns.boxplot(x='condition', y='retroClicksTotal', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax,'retroClicksAvg',y_label='average number of clicks', title='Pauses of the Video in the analysis task')

avg_conf = None
for i in range(1,4):
    column_name = 'RetrospectionConf' + str(i) + '[predictConf1]'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
    show_and_save_plt(ax, os.path.join('single_agents',column_name), y_label='confidence',ylim=(1,7),
                      title='Confidence in the analysis of agent'+ str(i))

    if i == 1:
        avg_conf = data[column_name]
    else:
        avg_conf += data[column_name]


column_name = 'retroConfAvg'
data[column_name] = avg_conf / 3
ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'HL', 'R-S', 'HL-S'])
show_and_save_plt(ax, column_name, y_label='average confidence',
                  title='Average confidence in the analysis',ylim=(1,7))

data['retroTimeAvg'] = (data['retro1Time']+data['retro2Time']+data['retro3Time'])/3.0
ax = sns.boxplot(x='condition', y='retroTimeAvg', data=data, order=['R','HL','R-S','HL-S'])
show_and_save_plt(ax, 'trustTimeAvg', ylim=(0, 500))

