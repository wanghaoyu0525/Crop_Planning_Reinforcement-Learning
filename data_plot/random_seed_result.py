import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os.path


def average(m):
    return sum(m)/len(m)


def plot_result(reward, profit, rotation_p, plant_0_yeild):
    plt.figure()
    sns.set()
    plt.subplot(2, 2, 1)
    plt.plot(reward, label='reward')
    plt.subplot(2, 2, 2)
    plt.plot(profit, label='profit')
    plt.subplot(2, 2, 3)
    plt.plot(rotation_p, label='rotation_p')
    plt.subplot(2, 2, 4)
    plt.plot(plant_0_yeild, label='plant_0_yeild')
    #plt.savefig('C:\\Users\\fmh\\Desktop\\毕业设计\\DQN_crop_rotation_v8\\better_random_seed_plot\\plot_' + str(i) + ".png")
    plt.savefig('D:\\Python_program\\DQN_crop_rotation_v8\\better_random_seed_plot\\plot_' + str(i) + ".png")


#dirname = 'C:\\Users\\fmh\\Desktop\\毕业设计\\DQN_crop_rotation_v8\\data_with_random_seed\\data_'
dirname = 'D:\\Python_program\\DQN_crop_rotation_v8\\data_with_random_seed\\data_'

for i in range(25,126):
    filename = dirname + str(i) + '.csv'
    if not os.path.isfile(filename):
        # print('data_'+ str(i) + '.csv is missing!')
        continue
    reward = []
    profit = []
    rotation_p = []
    plant_0_yeild = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        j = 0
        for row in reader:
            if j != 0:
                reward.append(float(row[1]))
                profit.append(float(row[2]))
                rotation_p.append(float(row[3]))
                plant_0_yeild.append(float(row[4]))
            j += 1
    '''if average(reward[70:]) > 185 and min(reward[70:])> 160:
        if average(profit[70:]) > 35000 and min(profit[70:])> 20000:
            if average(rotation_p[70:]) < 20 and min(rotation_p[70:]) < 30:
                if average(plant_0_yeild[70:]) < 25 and min(plant_0_yeild[70:]) < 40:
                    print('——————————————————random seed',i,'is good!————————————————')'''
    # print('random seed '+ str(i) + ' done!')
    if average(plant_0_yeild[70:]) < 25 and min(plant_0_yeild[70:]) < 40:
        plot_result(reward, profit, rotation_p, plant_0_yeild)
        print('random seed ' + str(i) + ' done!')
print('all done!')