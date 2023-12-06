import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import numpy as np
from os.path import dirname
from price_data import decompose_GBDT_monthPredict
import Parameter

sns.set()

random_seed = range(45,66)
#WHY
dirname = dirname(os.path.abspath(__file__)).replace("data_plot", "outputs_dynamic/data_with_random_seed\\Multi_cooperative_Tune_month1_Pre_month1_Month0_data_")

p_reward = []
p_profit = []
p_rotation_p = []
p_plant_0_yeild = []
for i in random_seed:
    filename = dirname + str(i) + '.csv'
    reward = []
    profit = []
    rotation_p = []
    plant_0_yeild = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            reward.append(float(row[1]))
            profit.append(float(row[3]))
            rotation_p.append(float(row[5]))
            plant_0_yeild.append(float(row[6]))

    p_reward.append(np.array(reward))
    p_profit.append(np.array(profit))
    p_rotation_p.append(np.array(rotation_p))
    p_plant_0_yeild.append(np.array(plant_0_yeild))

rewards = np.concatenate(p_reward)
profits = np.concatenate(p_profit)
rotation_ps = np.concatenate(p_rotation_p)
plant_0_yeilds = np.concatenate(p_plant_0_yeild)

# episode=np.concatenate((range(len(p_reward[0])),range(len(p_reward[0])),range(len(p_reward[0])),range(len(p_reward[0])),range(len(p_reward[0])),range(len(p_reward[0])),range(len(p_reward[0]))))
num = len(p_reward)
l = len(p_reward[0])
episode=np.concatenate([range(l) for i in range(num)])
print(episode.shape)
print(rewards.shape)

# with sns.axes_style('white'):  # 使用white主题
with sns.axes_style('ticks'):  # 使用ticks主题
# with sns.axes_style('whitegrid'):  # 使用white主题
    # fig = plt.figure(figsize=(15, 15),facecolor='white')
    fig = plt.figure(facecolor='white')
    # plt.gca().set_facecolor('w')



    # fig.add_subplot(2, 2, 1)
    plt.tight_layout()
    sns.lineplot(x=episode,y=rewards)
    plt.xlabel("Episode",{'fontname':'Times New Roman','fontsize':20})
    plt.ylabel("Reward",{'fontname':'Times New Roman','fontsize':20})

    plt.figure(facecolor='white')
    # fig.add_subplot(2, 2, 2)
    plt.tight_layout()
    sns.lineplot(x=episode,y=profits)
    plt.xlabel("Episode",{'fontname':'Times New Roman','fontsize':20})
    plt.ylabel("Profit(10^3CNY)",{'fontname':'Times New Roman','fontsize':20})

    plt.figure(facecolor='white')
    # fig.add_subplot(2, 2, 3)
    plt.tight_layout()
    sns.lineplot(x=episode,y=rotation_ps)
    plt.xlabel("Episode",{'fontname':'Times New Roman','fontsize':20})
    plt.ylabel("Rotation problem",{'fontname':'Times New Roman','fontsize':20})

    plt.figure(facecolor='white')
    # fig.add_subplot(2, 2, 4)
    plt.tight_layout()
    sns.lineplot(x=episode,y=plant_0_yeilds)
    plt.xlabel("Episode",{'fontname':'Times New Roman','fontsize':20})
    plt.ylabel("Planting at an inappropriate time",{'fontname':'Times New Roman','fontsize':20})

plt.show()