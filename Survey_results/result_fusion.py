import pandas as pd
import numpy as np
import os

delete_colums = ['startlanguage', 'submitdate','videoname1','videoname2','videoname3','retrospectionText','retrospectionQuest',
                 'next','infoPacman','gamePacman','regularPill','identifyPacman','powerPill','GhostsBlueWhen','GhostsBlueWhat',
                 'infoPacman2','gamePacman2','infoSummary','infoSaliency','greenArea','brightnessArea','quizSummary',
                 'infoSummary2','infoSaliency2', 'retrospectionVideo2','retrospectionVideo3','retrospectionVideo1',
                 'trustDescription2', 'videos2', 'trustDescription3', 'videos3','trustDescription1', 'videos1','submit'
                 ]
keep_time_colums=['demographicsTime','retro1Time', 'retro2Time', 'retro3Time', 'trust1Time', 'trust2Time', 'trust3Time']

def rename_time(data_frame, successive_name, new_name):
    index = data_frame.columns.get_loc(successive_name)
    index -= 1
    old_column_name = data_frame.columns[index]
    data_frame = data_frame.rename(columns={old_column_name: new_name}, errors="raise")
    return data_frame


def rename_times(data_frame):
    data_frame = rename_time(data_frame, 'ageTime', 'demographicsTime')
    data_frame = rename_time(data_frame, 'retrospectionVideo1Time', 'retro1Time')
    data_frame = rename_time(data_frame, 'retrospectionVideo2Time', 'retro2Time')
    data_frame = rename_time(data_frame, 'retrospectionVideo3Time', 'retro3Time')
    data_frame = rename_time(data_frame, 'trustDescription1Time', 'trust1Time')
    data_frame = rename_time(data_frame, 'trustDescription2Time', 'trust2Time')
    data_frame = rename_time(data_frame, 'trustDescription3Time', 'trust3Time')
    return data_frame

def delete_times(data_frame, keep_time_columns):
    start_index = data_frame.columns.get_loc('interviewtime') + 1
    to_delete = []
    for i in range(start_index,len(data_frame.columns)):
        column_name = data_frame.columns[i]
        if column_name not in keep_time_colums:
            to_delete.append(column_name)
    data_frame = data_frame.drop(to_delete, axis=1)
    return data_frame

def eval_retrospection(data_frame, number):
    def get_name(index):
        return 'retrospection' + str(number) + '[' + str(index) + ']'
    pacman = data_frame[get_name(1)].tolist()
    normal_pill = data_frame[get_name(2)].tolist()
    power_pill = data_frame[get_name(3)].tolist()
    ghost = data_frame[get_name(4)].tolist()
    blue_ghost = data_frame[get_name(5)].tolist()
    cherry = data_frame[get_name(6)].tolist()

    length = len(pacman)

    #define point calculation
    def calculate_points(index):
        i = index
        #check if they selected to many items
        if pacman[i] + normal_pill[i] + power_pill[i] + ghost[i] + blue_ghost[i] + cherry[i] > 3:
            points = 0
        else:
            if number == 1:
                points = pacman[i] + normal_pill[i] * -1 + power_pill[i] + ghost[i] * -1 + blue_ghost[i] * -1 + cherry[i] * -1
            elif number == 2:
                points = pacman[i] + normal_pill[i] * -0.5 + power_pill[i] * -0.5 + ghost[i] * -0.5 + blue_ghost[i] * 1 + cherry[i] * -0.5
            elif number == 3:
                points = pacman[i] + normal_pill[i] * -0.5 + power_pill[i] * -0.5 + ghost[i] * 1 + blue_ghost[i] * 1 + cherry[i] * -0.5
            else:
                print('number', number, 'not implemented')
                points = 'Nan'
        return points

    points = []
    for j in range(length):
        points.append(calculate_points(j))

    column_name = 'retrospection' + str(number) + 'Points'
    data_frame[column_name] = points


    # did they get the goal
    def got_goal(index, number):
        i = index
        # check if they selected to many items
        if pacman[i] + normal_pill[i] + power_pill[i] + ghost[i] + blue_ghost[i] + cherry[i] > 3:
            points = 0
        else:
            if number == 1:
                points = power_pill[i]
            elif number == 2:
                points = blue_ghost[i]
            elif number == 3:
                points = ghost[i]
            else:
                print('number', number, 'not implemented')

        return points

    goals = []
    for j in range(length):
        goals.append(got_goal(j,number))

    column_name = 'retrospection' + str(number) + 'Goal'
    data_frame[column_name] = goals

    return data_frame


def eval_trust(data_frame, agent_number):
    correct_answers_dict = {1:2,2:1, 3:1} #the correct answer for each comparison
    column_name = 'trustBool' + str(agent_number)
    resulting_column_name = 'trust' + str(agent_number) + 'correct'
    correct_answer_arr = []
    answer_arr = data_frame[column_name]
    for entry in answer_arr:
        if entry == correct_answers_dict[agent_number]:
            correct = 1;
        else:
            correct = 0;
        correct_answer_arr.append(correct)

    data_frame[resulting_column_name] = correct_answer_arr

    return data_frame








files = []
file_names = ['results_all_grps.csv','results_grp3.csv','results_grp4.csv']
for file_name in file_names:
    data_frame = pd.read_csv(file_name, sep=',')
    data_frame = rename_times(data_frame)
    files.append(data_frame)

resulting_frame = pd.concat(files, axis=0, ignore_index=True, sort=False)

resulting_frame.to_csv('raw_fusion.csv')

resulting_frame=resulting_frame[resulting_frame.lastpage == 17]

#resulting_frame.to_csv('fusion_finishedsurvey.csv')

resulting_frame = resulting_frame.drop(delete_colums, axis=1)
resulting_frame = delete_times(resulting_frame,keep_time_colums)

#resulting_frame.to_csv('fusion_cleaned.csv')

for i in range(1,4):
    resulting_frame = eval_retrospection(resulting_frame,i)

#resulting_frame.to_csv('fusion_retrospection_eval.csv')

for i in range(1,4):
    resulting_frame = eval_trust(resulting_frame,i)

resulting_frame.to_csv('fusion_final.csv')



