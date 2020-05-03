"""
Evaluates the main values except for the textual information.
Defines some functions used by the other evaluation scripts.
Most importantly, the functions need for the mann_whitney tests and the *show_and_save_plt* function that
defines how all plots look like.
"""

import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os


def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""

    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]


def bootstrap_mean(x, B=100000, alpha=0.05, plot=False):
    """Bootstrap standard error and (1-alpha)*100% c.i. for the population mean

    Returns bootstrapped standard error and different types of confidence intervals"""

    # Deterministic things
    n = len(x)  # sample size

    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)

    quantile_boot = np.percentile(sampling_distribution, q=(100*alpha/2, 100*(1-alpha/2)))

    if plot:
        plt.hist(sampling_distribution, bins="fd")
    return quantile_boot


def mean_and_CI(values):
    print('mean: ', values.mean())
    print(bootstrap_mean(values))


def rank_biserial_effect_size(x,y):
    mann_whitney_res = stats.mannwhitneyu(x, y)
    u = mann_whitney_res[0]
    effect_size = 1.0-((2.0*u)/(len(x)*len(y)))
    return effect_size


def mann_whitney(data, column_name, mean_only = False):
    ''' calculating either the mean or the mann whitney tests for for the dependant variable in the given column
    :param data: dataframe containing the data
    :param column_name: column of the dependent variable in the dataframe
    :param mean_only: if True, only the mean for each condition is calculated
    '''
    #splitting data into conditions
    data_random = data.loc[data.randnumber==1]
    data_randomLRP = data.loc[data.randnumber==3]
    data_highlights = data.loc[data.randnumber==2]
    data_highlightsLRP = data.loc[data.randnumber==4]

    random_values = data_random[column_name].values
    randomLRP_values = data_randomLRP[column_name].values
    highlights_values = data_highlights[column_name].values
    highlightsLRP_values = data_highlightsLRP[column_name].values

    print('##### testing the column:', column_name,'#####')

    #calculating mean and confidence intervall for each condition
    print('CI random:')
    mean_and_CI(random_values)
    print('CI random_LRP:')
    mean_and_CI(randomLRP_values)
    print('CI highlights:')
    mean_and_CI(highlights_values)
    print('CI highlights LRP:')
    mean_and_CI(highlightsLRP_values)

    #mann whitney tests
    if not mean_only:
        p_vals = []
        print('random < randomLRP')
        print(stats.mannwhitneyu(random_values, randomLRP_values, alternative='less'))
        print(rank_biserial_effect_size(random_values, randomLRP_values))
        p_vals.append(stats.mannwhitneyu(random_values, randomLRP_values, alternative='less')[1])
        print('highlights < highlightsLRP')
        print(stats.mannwhitneyu(highlights_values, highlightsLRP_values, alternative='less'))
        print(rank_biserial_effect_size(highlights_values, highlightsLRP_values))
        p_vals.append(stats.mannwhitneyu(highlights_values, highlightsLRP_values, alternative='less')[1])
        print('random < highlights')
        print(stats.mannwhitneyu(random_values, highlights_values, alternative='less'))
        print(rank_biserial_effect_size(random_values, highlights_values))
        p_vals.append(stats.mannwhitneyu(random_values, highlights_values, alternative='less')[1])
        print('randomLRP < highlightsLRP')
        print(stats.mannwhitneyu(randomLRP_values, highlightsLRP_values, alternative='less'))
        print(rank_biserial_effect_size(randomLRP_values, highlightsLRP_values))
        p_vals.append(stats.mannwhitneyu(randomLRP_values, highlightsLRP_values, alternative='less')[1])


def show_and_save_plt(ax ,file_name, y_label=None, title = None, ylim =None, label_size = 18, tick_size = 14):
    """
    Shows and saves the given plot and defines the appearance of the final plot.
    :param ax: the plot to be saved.
    :param file_name: save file name where the file is saved.
    :param y_label: the y axis label displayed
    :param title: titel of displayed in the plot (currently not used)
    :param ylim: limits of the y axis.
    :param label_size: font size of the label text
    :param tick_size: font size of the tick numbers
    """
    #this only works the second time the function is used, since it sets the style for future plots.
    # It was still more convenient this way. #TODO fix this
    sns.set_style("whitegrid")

    if y_label != None:
        plt.ylabel(y_label)
    plt.xlabel(None)
    #plt.supitle=(title)
    #ax.set(title=title)
    if ylim != None:
        ax.set(ylim=ylim)

    try:
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.label.set_size(label_size)
    except:
        try:
            plt.ylabel(y_label, fontsize=label_size)
            plt.xlabel(fontsize=label_size)
        except Exception as e:
            print(e)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    file_name = os.path.join('figures', file_name)
    if not (os.path.isdir(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    plt.tight_layout()
    plt.savefig(file_name)

    plt.show()


def satisfaction_analysis(number, data):
    '''
    Helper function to analyze the participants satisfaction in in each tast (Trust and Retrospection)
    :param number: specifies the Task: 1 = Retrospection, 2= Trust
    :param data: dataframe containing the data
    :return:
    '''

    # getting the correct column name for the given task number
    column_name = 'explSatisfaction' + str(number)
    if number == 1:
        result_name =  'explSatisfaction' + 'Retro' + 'Avg'
        title = 'Average satisfaction for the ' + 'retrospection task'
    elif number == 2:
        result_name = 'explSatisfaction' + 'Trust' + 'Avg'
        title = 'Average satisfaction for the ' + 'comparison task'
    else:
        print('number not implemented')
    # inverting the negative question 3 to be inline with the other positive questions
    data[column_name + '[3]'] = 6 - data[column_name + '[3]']

    # calculating the average satisfaction for all questions that were actually asked
    avg_satisfaction =[]
    for i in data.index.values:
        temp = 0
        if data['randnumber'][i] < 3:
            for j in (1,2,3,4):
                temp += data[column_name + '[' + str(j) + ']'][i]
            temp = temp / 4
        else:
            for j in (1, 2, 3,5,6):
                temp += data[column_name + '[' + str(j) + ']'][i]
            temp = temp / 5
        avg_satisfaction.append(temp)

    data[result_name] = avg_satisfaction
    ax = sns.barplot(x='condition', y=result_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, result_name, y_label='Average Rating', title=title, ylim=(1, 5))


def analyze_distribution(data, columns):
    """
    Analyzes the distribution of the values in the given columns
    :param data: The dataframe to be analyzed
    :param columns: The name of the columns to be compared
    :return ax: a plot showing the distribution
    """
    df = data[columns]
    def get_column_number(column):
        """
        Converts the column names to their position in the original columns-array.
        Should the name not be in the columns array the name will not be converted.
        """
        if column in columns:
            return columns.index(column) + 1
        else:
            return column

    df = pd.melt(df)
    df.variable = df.variable.apply(get_column_number)
    ax = sns.barplot(x="variable", y="value", data=df)

    return ax


if __name__ == '__main__':
    sns.set(palette='colorblind')

    ##### ANALYSIS OF TRUST TASK #####

    data = pd.read_csv('fusion_final.csv')
    data['condition'] = data.randnumber.apply(
        lambda x: 'R' if x == 1 else 'H' if x == 2 else 'R+S' if x == 3 else 'H+S')
    data.head()

    # remove participants that did not watch enough videos
    data['TrustClicksTotal'] = data.videoClicks1B + data.videoClicks1A + data.videoClicks2A + data.videoClicks2B + \
                               data.videoClicks3A + data.videoClicks3B
    total_number = len(data['TrustClicksTotal'])
    data = data.loc[data.TrustClicksTotal > 5]
    new_number = len(data['TrustClicksTotal'])
    difference = total_number - new_number
    print('did not watch:', difference)

    data['numTrustCorrect'] = data.trust1correct + data.trust2correct + data.trust3correct
    #data['percentTrustCorrect'] = data.numTrustCorrect.apply(lambda x: x/3.0)

    #for appropriate trust
    all_correct_values = []
    all_confidence_values = []
    all_conditions = []

    # generate plots for each agent
    for i in range(1,4):
        column_name = 'trust'+ str(i) +'correct'
        title = 'comparison task for agent ' + str(i)
        ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents',column_name), 'percentage of correct selections', title=title)

        #for appropriate trust
        all_correct_values.extend(data[column_name])

        column_name = 'trustConfidence' + str(i) + '[confidence1]'
        ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(1, 7))

        #for appropriate trust
        all_confidence_values.extend(data[column_name])
        all_conditions.extend(data['condition'])

        column_name = 'videoClicks' + str(i)
        data[column_name] = data[column_name+'A'] + data[column_name+'B']
        title = 'pauses during the comparison task for agent ' + str(i)
        ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents', column_name), 'number of pauses', title=title, ylim=(0, 10))

    # for appropriate trust
    appropriate_confidence = pd.DataFrame()
    appropriate_confidence['confidence'] = all_confidence_values
    appropriate_confidence['correct'] = all_correct_values
    appropriate_confidence['condition'] = all_conditions
    ax = sns.catplot(x="condition", y='confidence', hue="correct", kind="bar",
                     data=appropriate_confidence, order=['R', 'H', 'R+S', 'H+S'], legend=True, aspect=2);
    show_and_save_plt(ax, 'TrustAppropriateConfidence', y_label='Average Confidence',
                      ylim=(0, 7))

    # generate plots for all three agents combined
    ax = sns.barplot(x='condition', y='numTrustCorrect', data=data, order=['R','H','R+S','H+S'])
    show_and_save_plt(ax,'TrustPercentCorrect', y_label='Correct Agent Selections', title='combined trust tasks',
                      ylim=(0,3))

    satisfaction_analysis(2, data)

    column_name = 'TrustConfAvg'
    data[column_name] = (data['trustConfidence2[confidence1]']+data['trustConfidence3[confidence1]']+data['trustConfidence1[confidence1]'])/3.0
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, column_name, y_label='Average Confidence',
                      title='Average confidence in the agent selection',ylim=(1,7))

    data['trustTimeAvg'] = (data['trust1Time']+data['trust2Time']+data['trust3Time'])/3.0
    ax = sns.boxplot(x='condition', y='trustTimeAvg', data=data, order=['R','H','R+S','H+S'])
    show_and_save_plt(ax, 'trustTimeAvg',y_label='Seconds', ylim=(0, 500))

    data['TrustClicksAvg'] = data['TrustClicksTotal'] / 3
    ax = sns.boxplot(x='condition', y='TrustClicksAvg', data=data, order=['R','H','R+S','H+S'])
    show_and_save_plt(ax, 'TrustClicksAvg', y_label='Number of Pauses', title= 'Pauses of the Video in the trust task', ylim=(0, 10))

    # Mann whitney analyis
    mann_whitney(data, 'numTrustCorrect')
    mann_whitney(data, 'explSatisfaction' + 'Trust' + 'Avg')

    # analyze task specific satisfacttion
    data_No_LRP = data.loc[data.randnumber < 3]
    data_LRP = data.loc[data.randnumber > 2]

    number = 2
    base_column_name = 'explSatisfaction' + str(number)

    column_name = base_column_name + '[4]'
    ax = sns.barplot(x='condition', y=column_name, data=data_No_LRP, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'TrustSatisfactionVideo', y_label='Average Satisfaction', ylim=(1,5))

    column_name = base_column_name + '[5]'
    ax = sns.barplot(x='condition', y=column_name, data=data_LRP, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'TrustSatisfactionSummary', y_label='Average Satisfaction', ylim=(1,5)
                      )

    column_name = base_column_name + '[6]'
    ax = sns.barplot(x='condition', y=column_name, data=data_LRP, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'TrustSatisfactionHeatmap', y_label='Average Satisfaction', ylim=(1,5))

    column_name = base_column_name + '[3]'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'TrustSatisfactionTooMuch', y_label='Average Satisfaction', ylim=(1,5))

    # generate a dataframe of the main values, since it is easier to look at or use for future research
    # the main_values dataframes are not used in this project
    main_values = pd.DataFrame()
    main_values['seed'] = data.seed
    main_values['condition'] = data.condition
    main_values['explSatisfactionTrustAvg'] = data.explSatisfactionTrustAvg
    main_values['numTrustCorrect'] = data.numTrustCorrect

    main_values.to_csv('main_values_trust.csv')




    ############ ANALYSIS OF THE RETROSPECTION TASK ########
    # load data
    data = pd.read_csv('fusion_final.csv')
    data['condition'] = data.randnumber.apply(
        lambda x: 'R' if x == 1 else 'H' if x == 2 else 'R+S' if x == 3 else 'H+S')
    data.head()

    # remove participants that did not watch enough videos
    data['retroClicksTotal'] = data.retrospectionClicks1 + data.retrospectionClicks2 + data.retrospectionClicks3
    total_number = len(data['retroClicksTotal'])
    data = data.loc[data['retroClicksTotal'] > 2]
    new_number = len(data['retroClicksTotal'])
    difference = total_number - new_number
    print('did not watch Retrospection:', difference)

    # general values
    data['retroScoreTotal'] = (data['retrospection1Points']+ data['retrospection2Points']+ data['retrospection3Points'])
    data['retroGoalTotal'] = (data['retrospection1Goal']+ data['retrospection2Goal']+ data['retrospection3Goal'])
    data['retroPacmanTotal'] = (data['retrospection3[1]'] + data['retrospection3[1]'] + data['retrospection3[1]'])
    data['retroClicksTotal'] = data['retroClicksTotal'] / 3

    # for appropriate trust
    all_correct_values = []
    all_confidence_values = []
    all_conditions = []

    # generating graphs for each single agent
    for i in range(1,4):
        column_name = 'retrospection' + str(i) + 'Points'
        ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 1))

        column_name = 'retrospection' + str(i) + 'Goal'
        ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 1))

        # for appropriate trust
        all_correct_values.extend(data[column_name])

        column_name = 'retrospection' + str(i) + '[1]'
        ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents','retrospectionPacman'+str(i)), ylim=(0, 1))

        column_name = 'retrospectionClicks' + str(i)
        ax = sns.boxplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents',column_name), ylim=(0, 10))

    # generating graphs for all agents combined
    ax = sns.barplot(x='condition', y='retroScoreTotal', data=data, order=['R','H','R+S','H+S'])
    show_and_save_plt(ax,'retroScoreTotal', y_label= 'Total Score')

    ax = sns.barplot(x='condition', y='retroGoalTotal', data=data, order=['R','H','R+S','H+S'])
    show_and_save_plt(ax,'retroGoalTotal', y_label="Number of Selections",ylim=(0,3))

    ax = sns.barplot(x='condition', y='retroPacmanTotal', data=data, order=['R','H','R+S','H+S'])
    show_and_save_plt(ax,'retroPacmanTotal', y_label='Number of Selections', ylim=(0,3))

    ax = sns.boxplot(x='condition', y='retroClicksTotal', data=data, order=['R','H','R+S','H+S'])
    show_and_save_plt(ax,'retroClicksAvg',y_label='Number of Pauses', title='Pauses of the Video in the analysis task', ylim=(0, 10))

    data['retroTimeAvg'] = (data['retro1Time'] + data['retro2Time'] + data['retro3Time']) / 3.0
    ax = sns.boxplot(x='condition', y='retroTimeAvg', data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'retroTimeAvg', y_label='Seconds', ylim=(0, 500))

    # calculating confidence for each agent and the sum of confidence over all agents
    avg_conf = None
    for i in range(1,4):
        column_name = 'RetrospectionConf' + str(i) + '[predictConf1]'

        ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
        show_and_save_plt(ax, os.path.join('single_agents',column_name), y_label='confidence',ylim=(1,7),
                          title='Confidence in the analysis of agent'+ str(i))

        # for appropriate trust
        all_confidence_values.extend(data[column_name])
        all_conditions.extend(data['condition'])

        if i == 1:
            avg_conf = data[column_name]
        else:
            avg_conf += data[column_name]

    # for appropriate trust
    appropriate_confidence = pd.DataFrame()
    appropriate_confidence['confidence'] = all_confidence_values
    appropriate_confidence['correct'] = all_correct_values
    appropriate_confidence['condition'] = all_conditions
    ax = sns.catplot(x="condition", y='confidence', hue="correct", kind="bar",
                     data=appropriate_confidence, order=['R', 'H', 'R+S', 'H+S'], legend=True, aspect=2);
    show_and_save_plt(ax, 'GoalAppropriateConfidence', y_label='Average Confidence',
                      ylim=(0, 7))

    # generating graph for average confidence
    column_name = 'retroConfAvg'
    data[column_name] = avg_conf / 3
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, column_name, y_label='Average Confidence',
                      title='Average confidence in the analysis',ylim=(1,7))

    # general satisfaction analysis (same as for Trust task)
    satisfaction_analysis(1,data)

    # Mann Whiteny tests
    mann_whitney(data, 'retroScoreTotal')
    mann_whitney(data, 'explSatisfaction' + 'Retro' + 'Avg')

    mann_whitney(data, 'retroGoalTotal', mean_only=True)
    mann_whitney(data, 'retroPacmanTotal', mean_only=True)


    # analyze task specific satisfacttion
    data_No_LRP = data.loc[data.randnumber < 3]
    data_LRP = data.loc[data.randnumber > 2]

    number = 1
    base_column_name = 'explSatisfaction' + str(number)

    column_name = base_column_name + '[4]'
    ax = sns.barplot(x='condition', y=column_name, data=data_No_LRP, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'retroSatisfactionVideo', y_label='Average Satisfaction', ylim=(1,5))

    column_name = base_column_name + '[5]'
    ax = sns.barplot(x='condition', y=column_name, data=data_LRP, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'retroSatisfactionSummary', y_label='Average Satisfaction', ylim=(1,5)
                      )

    column_name = base_column_name + '[6]'
    ax = sns.barplot(x='condition', y=column_name, data=data_LRP, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'retroSatisfactionHeatmap', y_label='Average Satisfaction',ylim=(1,5))

    column_name = base_column_name + '[3]'
    ax = sns.barplot(x='condition', y=column_name, data=data, order=['R', 'H', 'R+S', 'H+S'])
    show_and_save_plt(ax, 'retroSatisfactionTooMuch', y_label='Average Satisfaction',ylim=(1,5))

    #### analye distribution of chosen objects:

    def sum_retro_item(idx):
        idx = '[' + str(idx) + ']'
        sum = data['retrospection'+str(1)+ idx]
        for agent in range(3,4):
             sum += data['retrospection'+str(agent)+ idx]
        return sum

    data['pacman'] = sum_retro_item(1)
    data['normal_pill'] = sum_retro_item(2)
    data['power_pill'] = sum_retro_item(3)
    data['ghost'] = sum_retro_item(4)
    data['blue_ghost'] = sum_retro_item(5)
    data['cherry'] = sum_retro_item(6)

    data_random = data.loc[data.randnumber == 1]
    ax = analyze_distribution(data_random,
                              ['pacman', 'normal_pill', 'power_pill', 'ghost', 'blue_ghost', 'cherry'])
    show_and_save_plt(ax, os.path.join('single_agents','retroDistributionR'))

    data_highlights = data.loc[data.randnumber == 2]
    ax = analyze_distribution(data_highlights,
                              ['pacman', 'normal_pill', 'power_pill', 'ghost', 'blue_ghost', 'cherry'])
    show_and_save_plt(ax, os.path.join('single_agents','retroDistributionH'))

    data_randomLRP = data.loc[data.randnumber == 3]
    ax = analyze_distribution(data_randomLRP,
                              ['pacman', 'normal_pill', 'power_pill', 'ghost', 'blue_ghost', 'cherry'])
    show_and_save_plt(ax, os.path.join('single_agents','retroDistributionRS'))

    data_highlightsLRP = data.loc[data.randnumber == 4]
    ax = analyze_distribution(data_highlightsLRP, ['pacman','normal_pill', 'power_pill','ghost','blue_ghost','cherry'])
    show_and_save_plt(ax, os.path.join('single_agents','retroDistributionHS'))

    # generate a dataframe of the main values, since it is easier to look at or use for future research
    # the main_values dataframes are not used in this project
    main_values = pd.DataFrame()
    main_values['seed'] = data.seed
    main_values['condition'] = data.condition
    main_values['explSatisfactionRetroAvg'] = data.explSatisfactionRetroAvg
    main_values['retroScoreTotal'] = data.retroScoreTotal

    main_values.to_csv('main_values_retrospection.csv')