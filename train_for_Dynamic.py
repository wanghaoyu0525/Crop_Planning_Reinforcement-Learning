import sys, os
import torch
import datetime
import random
import numpy as np
from common.utils import save_results, make_dir
from common.plot import plot_rewards, plot_supply
from agent import DQN
import pandas as pd
import argparse
import csv
import Parameter
import matplotlib.pyplot as plt

#WHY
import time
import sys



from torch.utils.tensorboard import SummaryWriter

from environment.env_v5_why_GBDTPrice import MyEnv
from environment.farmer import farmer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


# 设置随机数种子
# setup_seed(35)

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DQNConfig:
    def __init__(self):
        self.algo = "DQN"  # 算法名称
        self.env = 'crop planning'  # 环境名称
        os.makedirs(str(curr_path + './outputs_dynamic/' + self.env), exist_ok=True)
        if Parameter.Tune_month_OR_Pre_month == 0:
            self.result_path = curr_path + "/outputs_dynamic/" + self.env + \
                           '/Multi_coperative' +str(Parameter.Save_name) +'/'+\
                               str(Parameter.Tune_month[0]) + 'Tune_month_'+str(Parameter.Pre_month[0]) \
                               + 'Pre_month_'+ str(Parameter.Num_Month) + "Month"  + '/results/'  # 保存结果的路径
            #'/' + str(Parameter.Num_Month) + "_Month_" + curr_time + '/results/'  # 保存结果的路径
            self.model_path = curr_path + "/outputs_dynamic/" + self.env + \
                              '/Multi_coperative' +str(Parameter.Save_name) +'/'+\
                              str(Parameter.Tune_month[0]) + 'Tune_month_'+str(Parameter.Pre_month[0]) \
                              + 'Pre_month_'+ str(Parameter.Num_Month) + "Month" + '/models/'  # 保存模型的路径
             #'/' + str(Parameter.Num_Month) + "_Month_" + curr_time + '/models/'  # 保存模型的路径
        elif Parameter.Tune_month_OR_Pre_month == 1:
            self.result_path = curr_path + "/outputs_dynamic/" + self.env + \
                               '/Multi_Tune_month' +str(Parameter.Save_name)+'/' +\
                               str(Parameter.Pre_month[0]) + 'Pre_month_'+ str(Parameter.Num_Month) + \
                               "Month" + '/results/'  # 保存结果的路径
            self.model_path = curr_path + "/outputs_dynamic/" + self.env + \
                              '/Multi_Tune_month' +str(Parameter.Save_name)+'/' +\
                              str(Parameter.Pre_month[0]) + 'Pre_month_'+ str(Parameter.Num_Month) + \
                              "Month" + '/models/'  # 保存模型的路径
        elif Parameter.Tune_month_OR_Pre_month == 2:
            self.result_path = curr_path + "/outputs_dynamic/" + self.env + \
                               '/Multi_Pre_month' + str(Parameter.Save_name)+'/' +str(Parameter.Tune_month[0])\
                               + 'Tune_month_'+ str(Parameter.Num_Month) + "Month" + '/results/'  # 保存结果的路径
            self.model_path = curr_path + "/outputs_dynamic/" + self.env + \
                              '/Multi_Pre_month' + str(Parameter.Save_name)+'/' +str(Parameter.Tune_month[0]) \
                              + 'Tune_month_'+ str(Parameter.Num_Month) + "Month" + '/models/'
        elif Parameter.Tune_month_OR_Pre_month == 3:
            self.result_path = curr_path + "/outputs_dynamic/" + self.env + \
                               '/Multi_Price' + str(Parameter.Save_name)+'/' +str(Parameter.Tune_month[0])\
                               + 'Tune_month_'+ str(Parameter.Pre_month[0]) + 'Pre_month_' + str(Parameter.Num_Month) + "Month" + '/results/'  # 保存结果的路径
            self.model_path = curr_path + "/outputs_dynamic/" + self.env + \
                              '/Multi_Price' + str(Parameter.Save_name)+'/' +str(Parameter.Tune_month[0]) \
                              + 'Tune_month_'+ str(Parameter.Pre_month[0]) + 'Pre_month_' + str(Parameter.Num_Month) + "Month" + '/models/'
        else:
            self.result_path = curr_path + "/outputs_dynamic/" + self.env + \
                               '/' + str(Parameter.Save_name) +'/'+ str(Parameter.Tune_month[0]) + \
                               'Tune_month_'+str(Parameter.Pre_month[0]) + 'Pre_month_'+str(
                Parameter.Num_Month) + "Month" + '/results/'  # 保存结果的路径
            self.model_path = curr_path + "/outputs_dynamic/" + self.env + \
                              '/' + str(Parameter.Save_name) +'/'+ str(Parameter.Tune_month[0]) + \
                              'Tune_month_'+str(Parameter.Pre_month[0]) + 'Pre_month_'+str(
                Parameter.Num_Month) + "Month" + '/models/'
        self.train_eps = Parameter.Train_Eps  # 训练的回合数
        self.eval_eps = 10  # 测试的回合数
        self.gamma = 0.98  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.001  # 学习率
        self.memory_capacity = 1000000  # 经验回放的容量
        self.batch_size = 256  # mini-batch SGD中的批量大小
        self.target_update = 4  # 目标网络的更新频率
        self.tau = 0.005
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.hidden_dim = 256  # hidden size of net

        # 将所有农户定义面积后放入一个集合中，农户的面积是[1,2)之间的浮点数
        self.farmer_all = []
        for i in range(Parameter.num_farmer):
            #self.farmer_all.append(farmer(i, random.random() + 1))
            self.farmer_all.append(farmer(i, random.random() + 1, random.randint(1,13)))

    def get_farmer_num(self):
        return len(self.farmer_all)


def env_agent_config(cfg, seed=1):
    init_state = []
    for i in range(len(cfg.farmer_all)):
        init_state.append(random.randint(1, 13))
    env = MyEnv(cfg.farmer_all, init_state)
    if Parameter.ConsiderMiniSupply == True:

        n_states = Parameter.n_states
        # n_states = env.num_crop * 4 + 1 + 4
        #n_states = env.num_crop * 4 + 1 + 3
        # n_states = env.num_crop * 5 + 1 + 4
        # num = 0
        # for crop in range(1, Parameter.num_crop + 1):
        #     if Parameter.Min_Demand[crop] > 0:
        #         num += 1
        # n_states = Parameter.num_crop * 4 + 1 + 4 + num  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
        #n_states = Parameter.num_crop * 3 + 1 + 4 + num * 2 # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；


        #n_states = env.num_crop * 4 + 1 + 6
    else:
        n_states = Parameter.n_states
        # n_states = env.num_crop * 3 + 1 + 3
        #n_states = env.num_crop * 3 + 1 + 6
    n_actions = Parameter.n_actions
    # n_actions = env.num_crop + 1
    agent = DQN(n_states, n_actions, cfg)
    return env, agent



def train_Dynamic(cfg, env, agent, num_month,num_cooperate):
    #time_start = time.time()  #WHY 记录开始时间
    time_start = time.perf_counter()  # 记录开始时间
    print('开始训练!')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励

    total_iteration = 0  # 记录环境step的次数
    total_update = 0  # 记录环境step的次数
    if  Parameter.Tune_month_OR_Pre_month != 3:
        num_cooperate = 0 #价格没有变化
    if Parameter.UpdatedPredictPrice == False:  # 不动态更新价格预测算法，根据确定好的算法更新价格列表
        if Parameter.Step_or_data == 1:
            env.Update_PriceList(Parameter.Num_Month,0,num_cooperate)  # 价格准备好,预测月数据用默认的参数0即可
        elif Parameter.Step_or_data == 0:
            env.Update_PriceList(Parameter.Num_Month, Parameter.Max_Step-Parameter.Num_Month+Parameter.Future_Month,num_cooperate)  # 价格准备好
    else:
        if Parameter.Step_or_data == 1:
            env.MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month , num_month,
                                            num_cooperate,True)  # 需要从第Parameter.Num_Month个月开始预测
        elif Parameter.Step_or_data == 0:
            env.MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month, num_month,
                                            num_cooperate,True)  # 需要从第Parameter.Num_Month个月开始预测

    for i_ep in range(cfg.train_eps):
        obs = env.reset()
        ep_reward = 0
        #profit = []

        profit_month = []  # 输出每个月的实际利润
        profit_12_month = []  # 输出过去12个月的月均利润
        profit_total = []  # 输出截止到当月的总利润


        rotation_problem = 0
        plant_0_yeild = 0
        action_0 = 0
        MiniSupply_problem = 0
        MiniSupply_problem_list = [0.0 for i in range(Parameter.num_crop + 1)]
        supply = [0 for i in range(Parameter.num_crop)]

        step_num = 0
        # time_start_ep = time.time()  #WHY  记录回合开始时间
        time_start_ep = time.perf_counter()  # 记录开始时间
        action_choose_ep = [0.0 for i in range(Parameter.num_crop + 1)]
        while True:

            action = []
            memory_in_one_month = {}
            memory_in_one_month['state'] = []
            memory_in_one_month['action'] = []
            memory_in_one_month['next_state'] = []
            for f in cfg.farmer_all:
                state = []
                state.extend(obs)
                # 归一化操作
                s1 = f.area * 2 - 3
                s2 = f.current_crop * 2 / 13 - 1#作物
                s3 = f.plant_month * 2 / 11 - 1#种植时间
                if Parameter.ConsiderMiniSupply == True:
                    f_total_yeild = sum(Parameter.yeild[f.current_crop][f.plant_month % 12]) + 1
                    s4 = f_total_yeild * 0.5 - 1.25#产量
                # f_harvest_time = Parameter.yeild[f.current_crop][f.plant_month % 12].index(-1) - 1
                # s5 = ((f_harvest_time + 12 - f.plant_month) % 12 - 5) / 3#种植生育期间隔
                # inteval_month = f_harvest_time - f.plant_month
                # if inteval_month < 0:
                #     inteval_month += 12
                # if f.current_crop > 0:
                #     s6 = (env.price[f.current_crop - 1][(f.plant_month + inteval_month - 1)] * 10 - 47) / 40  # 收获月份价格
                # else:
                #     s6 = (0 * 10 - 47) / 40  # 收获月份价格
                # state.extend([s1, s2, s3, s4, s5, s6])
                if Parameter.ConsiderMiniSupply == True:
                    state.extend([s1, s2, s3, s4])
                else:
                    state.extend([s1, s2, s3])


                if Parameter.ConsiderMiniSupply == True:  # 考虑最小供给，把剩余的需求放入状态
                    # # 个人对合作社的贡献
                    # Obs = []
                    # for crop in range(1, Parameter.num_crop + 1):
                    #     if env.min_demand[crop] <= 0:
                    #         Obs.append(1)
                    #     else:
                    #     #if env.min_demand[crop] > 0:
                    #         if Parameter.YearOrMonthSupply == False:  # 如果是稳定按月供给，这块没仔细考虑！！！
                    #             m_d = 1 * (env.min_demand[crop] - sum(env.supply_in_last_12[crop])) / Parameter.num_farmer
                    #             last_demand = m_d - sum(f.supply_in_last_12[crop])
                    #             if last_demand >= 0:
                    #                 Obs.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    #             else:  # 需求满足
                    #                 Obs.append(1)
                    #         elif Parameter.YearOrMonthSupply == True:  # 如果是按照年供给
                    #             # assert (len(env.LastYearSupply[crop]) <= 12)
                    #             # m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop])) / Parameter.num_farmer
                    #             # last_demand = m_d - sum(f.LastYearSupply[crop])
                    #             # if last_demand >= 0:
                    #             #     Obs.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    #             # else:  # 需求满足
                    #             #     Obs.append(1)
                    #
                    #             # Future_supply = 0
                    #             # for m in range(1, 8):#最大生育期不超过8，要不把以前的又重复计算了
                    #             #     supply = f.area * Parameter.yeild[f.current_crop][f.plant_month][
                    #             #         (Parameter.Num_Month + m) % 12]
                    #             #     if supply >= 0:#去除等于-1的产量
                    #             #         Future_supply += supply
                    #
                    #             ratio = sum(f.LastYearSupply[crop])/ env.min_demand[crop]
                    #             if ratio > 1:
                    #                 ratio = 1
                    #             elif ratio < 0:
                    #                 ratio = 0
                    #             Obs.append(2 * ratio - 1)  # 最小需求是否满足 2x-1  -> -1 ~ 1
                    #
                    #             # m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop]))
                    #             # if m_d >= 0:
                    #             #     assert (env.min_demand[crop] > 0)
                    #             #     Obs.append((1 - 2 * m_d / env.min_demand[crop]))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    #             # else:  # 需求满足
                    #             #     Obs.append(1)
                    # state.extend(Obs)

                    #整个合作社的最小供给情况
                    Obs = []
                    for crop in range(1, Parameter.num_crop + 1):
                        if env.min_demand[crop] <= 0:
                            Obs.append(1)
                        else:
                            # if env.min_demand[crop] > 0:
                            if Parameter.YearOrMonthSupply == False:  # 如果是稳定按月供给，这块没仔细考虑！！！
                                m_d = 1 * (env.min_demand[crop] - sum(
                                    env.supply_in_last_12[crop])) / Parameter.num_farmer
                                last_demand = m_d - sum(f.supply_in_last_12[crop])
                                if last_demand >= 0:
                                    Obs.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                                else:  # 需求满足
                                    Obs.append(1)
                            elif Parameter.YearOrMonthSupply == True:  # 如果是按照年供给
                                # assert (len(env.LastYearSupply[crop]) <= 12)
                                # m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop])) / Parameter.num_farmer
                                # last_demand = m_d - sum(f.LastYearSupply[crop])
                                # if last_demand >= 0:
                                #     Obs.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                                # else:  # 需求满足
                                #     Obs.append(1)
                                m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop]))
                                if m_d >= 0:
                                    assert (env.min_demand[crop] > 0)
                                    Obs.append((1 - 2 * m_d / env.min_demand[crop]))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                                else:  # 需求满足
                                    Obs.append(1)
                    state.extend(Obs)
                    # #某种作物种植的户数占的总比
                    # area_ratio = [0 for i in range(Parameter.num_crop)]
                    # area_list = [0 for i in range(Parameter.num_crop)]
                    # area_total = 0
                    # for farmer in range(Parameter.num_farmer):
                    #     if env.farmer_all[farmer].current_crop > 0:
                    #         area_list[env.farmer_all[farmer].current_crop - 1] += env.farmer_all[farmer].area
                    #     area_total += env.farmer_all[farmer].area
                    # # for farmer in cfg.farmer_all:
                    # #     if farmer.current_crop > 0:
                    # #         area_list[farmer.current_crop - 1] += farmer.area
                    # for crop in range(0, Parameter.num_crop):
                    #     area_ratio[crop] = 2* (area_list[crop] / area_total) - 1
                    # state.extend(area_ratio)

                    # # 某种作物种植的户数占的总比
                    # area_ratio = []
                    # area_list = [0 for i in range(Parameter.num_crop)]
                    # area_total = 0
                    # for farmer in range(Parameter.num_farmer):
                    #     if env.farmer_all[farmer].current_crop > 0 and env.min_demand[
                    #         env.farmer_all[farmer].current_crop] > 0:
                    #         area_list[env.farmer_all[farmer].current_crop - 1] += env.farmer_all[farmer].area
                    #     area_total += env.farmer_all[farmer].area
                    # area_mini = []
                    # for crop in range(1, Parameter.num_crop + 1):
                    #     if env.min_demand[crop] > 0:
                    #         area_mini.append(area_list[crop - 1])
                    # for crop in range(len(area_mini)):
                    #     area_ratio.append(2 * (area_mini[crop] / area_total) - 1)
                    # state.extend(area_ratio)

                is_current_done = env.is_current_done(f.current_crop, f.plant_month)
                action_f = agent.choose_action(state, is_current_done)

                action_choose_ep[action_f] += 1

                memory_in_one_month['state'].append(state)
                memory_in_one_month['action'].append(action_f)
                if is_current_done:
                    s2 = action_f * 2 / 13 - 1
                    s3 = (env.current_month + 1) % 12 * 2 / 11 - 1 #WHY?current_month是从1，判断下个月？
                    if Parameter.ConsiderMiniSupply == True:
                        f_total_yeild = sum(Parameter.yeild[action_f][env.current_month % 12]) + 1
                        s4 = f_total_yeild * 0.5 - 1.25  # 产量
                    # #要满足的作物的总面积、单体产量是否满足需求
                    # crop_mini_demand = []
                    # for crop in range(1, Parameter.num_crop + 1):
                    #     # 任务分解到每个农户身上，乘以一个系数，因为不是每个人都正好有机会去种这个，所以要有冗余
                    #     if Parameter.min_demand[crop] > 0:
                    #         crop_mini_demand.append(crop)


                    # f_harvest_time = Parameter.yeild[action_f][ env.current_month% 12].index(-1) - 1
                    # s5 = ((f_harvest_time + 12 - env.current_month) % 12 - 5) / 3  # 种植生育期间隔
                    # inteval_month = f_harvest_time - env.current_month
                    # if inteval_month < 0:
                    #     inteval_month += 12
                    # if action_f > 0:
                    #     s6 = (env.price[action_f - 1][(env.current_month + inteval_month - 1)] * 10 - 47) / 40  # 收获月份价格
                    # else:
                    #     s6 = (0 * 10 - 47) / 40  # 收获月份价格

                #OBS = [s1, s2, s3, s4, s5, s6]
                #OBS = [s1, s2, s3]
                if Parameter.ConsiderMiniSupply == True:
                    OBS = [s1, s2, s3, s4]
                else:
                    OBS = [s1, s2, s3]
                if Parameter.ConsiderMiniSupply == True:  # 考虑最小供给，把剩余的需求放入状态
                    # 个人对合作社的贡献
                    # for crop in range(1, Parameter.num_crop + 1):
                    #     if env.min_demand[crop] <= 0:
                    #         OBS.append(1)
                    #     else:
                    #     #if env.min_demand[crop] > 0:
                    #         if Parameter.YearOrMonthSupply == False:  # 如果是稳定按月供给，这块没仔细考虑！！！
                    #             m_d = 1 * (env.min_demand[crop] - sum(env.supply_in_last_12[crop])) / Parameter.num_farmer
                    #             last_demand = m_d - sum(f.supply_in_last_12[crop])
                    #             if last_demand >= 0:
                    #                 OBS.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    #             else:  # 需求满足
                    #                 OBS.append(1)
                    #
                    #         elif Parameter.YearOrMonthSupply == True:  # 如果是按照年供给
                    #             # m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop])) / Parameter.num_farmer
                    #             # last_demand = m_d - sum(f.LastYearSupply[crop])
                    #             # if last_demand >= 0:
                    #             #     OBS.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    #             # else:  # 需求满足
                    #             #     OBS.append(1)
                    #
                    #             ratio = sum(f.LastYearSupply[crop]) / env.min_demand[crop]
                    #             if ratio > 1:
                    #                 ratio = 1
                    #             elif ratio < 0:
                    #                 ratio = 0
                    #             OBS.append(2 * ratio - 1)  # 最小需求是否满足 2x-1  -> -1 ~ 1
                    #
                    #             # m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop]))
                    #             # if m_d >= 0:
                    #             #     assert (env.min_demand[crop] > 0)
                    #             #     OBS.append((1 - 2 * m_d / env.min_demand[crop]))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    #             # else:  # 需求满足
                    #             #     OBS.append(1)
                    # 整个合作社的最小供给情况
                    for crop in range(1, Parameter.num_crop + 1):
                        if env.min_demand[crop] <= 0:
                            OBS.append(1)
                        else:
                        #if env.min_demand[crop] > 0:
                            if Parameter.YearOrMonthSupply == False:  # 如果是稳定按月供给，这块没仔细考虑！！！
                                m_d = 1 * (env.min_demand[crop] - sum(env.supply_in_last_12[crop])) / Parameter.num_farmer
                                last_demand = m_d - sum(f.supply_in_last_12[crop])
                                if last_demand >= 0:
                                    OBS.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                                else:  # 需求满足
                                    OBS.append(1)

                            elif Parameter.YearOrMonthSupply == True:  # 如果是按照年供给
                                # m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop])) / Parameter.num_farmer
                                # last_demand = m_d - sum(f.LastYearSupply[crop])
                                # if last_demand >= 0:
                                #     OBS.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                                # else:  # 需求满足
                                #     OBS.append(1)
                                m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop]))
                                if m_d >= 0:
                                    assert (env.min_demand[crop] > 0)
                                    OBS.append((1 - 2 * m_d / env.min_demand[crop]))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                                else:  # 需求满足
                                    OBS.append(1)


                    # # 某种作物种植的户数占的总比
                    # area_ratio = [0 for i in range(Parameter.num_crop)]
                    # area_list = [0 for i in range(Parameter.num_crop)]
                    # area_total = 0
                    # for farmer in range(Parameter.num_farmer):
                    #     if env.farmer_all[farmer].current_crop > 0:
                    #         area_list[env.farmer_all[farmer].current_crop - 1] += env.farmer_all[farmer].area
                    #     area_total += env.farmer_all[farmer].area
                    # # for farmer in cfg.farmer_all:
                    # #     if farmer.current_crop > 0:
                    # #         area_list[farmer.current_crop - 1] += farmer.area
                    # for crop in range(0, Parameter.num_crop):
                    #     area_ratio[crop] = 2 * (area_list[crop] / area_total) - 1
                    #     OBS.append(area_ratio[crop])

                    # # 某种作物种植的户数占的总比
                    # area_ratio = []
                    # area_list = [0 for i in range(Parameter.num_crop)]
                    # area_total = 0
                    # for farmer in range(Parameter.num_farmer):
                    #     if env.farmer_all[farmer].current_crop > 0 and env.min_demand[
                    #         env.farmer_all[farmer].current_crop] > 0:
                    #         area_list[env.farmer_all[farmer].current_crop - 1] += env.farmer_all[farmer].area
                    #     area_total += env.farmer_all[farmer].area
                    # area_mini = []
                    # for crop in range(1, Parameter.num_crop + 1):
                    #     if env.min_demand[crop] > 0:
                    #         area_mini.append(area_list[crop - 1])
                    # for crop in range(len(area_mini)):
                    #     OBS.append(2 * (area_mini[crop] / area_total) - 1)


                memory_in_one_month['next_state'].append(OBS)
                action.append(action_f)



            obs, reward, done, info = env.step(action)#WHY?done一直为false
            step_num += 1
            # print("step_num is " + str(step_num) )
            #WHY
            for i in range(len(memory_in_one_month['state'])):
                next_s = []
                next_s.extend(obs)
                next_s.extend(memory_in_one_month['next_state'][i])
                agent.memory.push(memory_in_one_month['state'][i], memory_in_one_month['action'][i], reward[i], next_s,
                                  done)

            ep_reward += sum(reward) / len(reward)#WHY所有农户平均回报
            #profit.append(info['profit'])#WHY过去12个月所有农户平均收益

            # 输出每个月的实际利润
            profit_month.append(round(info['actual_profit'], 2))
            # 输出过去12个月的月均利润
            profit_12_month.append(round(info['profit'], 2))
            # 输出截止到当月的总利润
            if (len(profit_total) == 0):
                profit_total.append(round(info['actual_profit'], 2) + 0)
            else:
                profit_total.append(profit_total[-1] + round(info['actual_profit'], 2))


            rotation_problem += info['rotation_problem']
            plant_0_yeild += info['plant_0_yeild']
            action_0 += info['action_0']
            MiniSupply_problem += info['MiniSupply_problem']
            for i in range(Parameter.num_crop + 1):
                MiniSupply_problem_list[i] += info['MiniSupply_problem_list'][i]

            for crop in range(len(supply)):
                supply[crop] += info['supply'][crop + 1]#WHY实际作物供给
            # state = next_state
            #if done or step_num > Parameter.Train_Eps:
            #if done or step_num > 200:
            #max_ = max(Parameter.Future_Month, num_month)
            #if done or step_num > Parameter.Num_Month + max_ - 1:
            #if done or step_num > Parameter.Num_Month + Parameter.Future_Month - 1:
            if Parameter.Step_or_data == 1:
                if done or step_num > Parameter.Num_Month:
                    break
            elif Parameter.Step_or_data == 0:
                if done or step_num > Parameter.Max_Step:
                    break

            total_iteration += 1

            if total_iteration >= cfg.batch_size * cfg.get_farmer_num() * 1:
                #  farmer_num = 100
                for i in range(10):  # 训练十次
                    agent.update()  # 更新网络
                    total_update += 1

        if (i_ep + 1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        # 改变成使用目标网络平滑的方式
        if (i_ep + 1) % 5 == 0:
            print('回合：{}/{}, 奖励：{}, 累积step次数: {}, 累计更新次数: {}'.format(i_ep + 1, cfg.train_eps, ep_reward,
                                                                                   total_iteration, total_update))

            #time_end_ep = time.time()  #WHY 记录回合结束时间
            time_end_ep = time.perf_counter()  # 记录开始时间
            time_sum = time_end_ep - time_start_ep  # 计算的时间差为程序的执行时间，单位为秒/s
            print('回合运行时间为', time_sum)
            # print('回合：{}/{}, 奖励：{}, 更新次数: {}'.format(i_ep + 1, cfg.train_eps, ep_reward,env.get_counter()))
            # writer.add_scalar('reward+', ep_reward, i_ep)
            # writer.add_scalar('profit in last year+', sum(profit), i_ep)
            # writer.add_scalar('rotation problem number-',rotation_problem,i_ep)
            # writer.add_scalar('Plant at an inappropriate time-', plant_0_yeild, i_ep)
            # writer.add_scalar('Plant nothing-', action_0, i_ep)
            # print(supply)
        rewards.append(ep_reward)
        os.makedirs('./outputs_dynamic/data_with_random_seed', exist_ok=True)
        os.makedirs('./outputs_dynamic/supply_with_random_seed', exist_ok=True)
        if Parameter.Tune_month_OR_Pre_month == 1:
            filename_w = './outputs_dynamic/data_with_random_seed/Multi_Tune_month'+ str(Parameter.Save_name) +'_Pre_month'+str(Parameter.Pre_month[0]) + '_Month'+ str(Parameter.Num_Month) + '_data_' + str(
                Parameter.Random_Seed) + '.csv'
            filename_supply = './outputs_dynamic/supply_with_random_seed/Multi_Tune_month'+str(Parameter.Save_name) + '_Pre_month'+str(Parameter.Pre_month[0]) +'_Month'+ str(
                Parameter.Num_Month) + '_data_' + str(Parameter.Random_Seed) + '.csv'

        elif Parameter.Tune_month_OR_Pre_month == 2:
            filename_w = './outputs_dynamic/data_with_random_seed/Multi_Pre_month' + str(Parameter.Save_name) + '_Tune_month'+ str(Parameter.Tune_month[0]) + \
                               '_Month'+ str(Parameter.Num_Month) + '_data_' + str(Parameter.Random_Seed) + '.csv'
            filename_supply = './outputs_dynamic/supply_with_random_seed/Multi_Pre_month' + str(Parameter.Save_name) + '_Tune_month'+str(Parameter.Tune_month[0]) + \
                               '_Month'+ str(
                Parameter.Num_Month) + '_data_' + str(Parameter.Random_Seed) + '.csv'

        elif Parameter.Tune_month_OR_Pre_month == 3:
            filename_w = './outputs_dynamic/data_with_random_seed/Multi_Price' + str(Parameter.Save_name) + '_Tune_month'+ str(Parameter.Tune_month[0]) + \
                               '_Pre_month'+str(Parameter.Pre_month[0])+'_Month'+ str(Parameter.Num_Month) + '_data_' + str(Parameter.Random_Seed) + '.csv'
            filename_supply = './outputs_dynamic/supply_with_random_seed/Multi_Price' + str(Parameter.Save_name) + '_Tune_month'+str(Parameter.Tune_month[0]) + \
                               '_Pre_month'+str(Parameter.Pre_month[0])+'_Month'+ str(
                Parameter.Num_Month) + '_data_' + str(Parameter.Random_Seed) + '.csv'

        else:
            filename_w = './outputs_dynamic/data_with_random_seed/Multi_cooperative'+ str(Parameter.Save_name) + '_Tune_month'+ str(Parameter.Tune_month[0])+'_Pre_month'+str(Parameter.Pre_month[0]) +'_Month'+ str(Parameter.Num_Month) + '_data_' + str(Parameter.Random_Seed) + '.csv'
            filename_supply = './outputs_dynamic/supply_with_random_seed/Multi_cooperative'+ str(Parameter.Save_name) + '_Tune_month'+ str(Parameter.Tune_month[0])+'_Pre_month'+str(Parameter.Pre_month[0]) +'_Month'+ str(Parameter.Num_Month) + '_data_' + str(Parameter.Random_Seed) + '.csv'
        #WHY：第几轮次，人均回报，输出每个月的实际利润, 输出过去12个月的月均利润,输出截止到当月的总利润，轮更问题，产量为0
        #result = pd.DataFrame([[i_ep, ep_reward, sum(profit), rotation_problem, plant_0_yeild]],
        # result = pd.DataFrame([[i_ep, ep_reward, sum(profit_month), sum(profit_12_month), sum(profit_total), rotation_problem, plant_0_yeild, MiniSupply_problem]],
        #                       columns=['i_ep', 'reward', 'profit_month', 'profit_12past_month','profit_total', 'rotation_p', 'plant_0_yeild','MiniSupply_problem'])

        data_1 = [i_ep, ep_reward, sum(profit_month), sum(profit_12_month), sum(profit_total), rotation_problem, plant_0_yeild]
        name = ['i_ep', 'reward', 'profit_month', 'profit_12past_month', 'profit_total', 'rotation_p', 'plant_0_yeild']
        for i in range(Parameter.num_crop):
            data_1.append(MiniSupply_problem_list[i+1])
            name.append(Parameter.action_name[i+1]+ '_notsatisify')
        for i in range(Parameter.num_crop + 1):
            data_1.append(action_choose_ep[i])
            name.append(str(Parameter.action_name[i]) + '_choose')

        result = pd.DataFrame([data_1],columns=name)

        supply.insert(0, i_ep)
        result_supply = pd.DataFrame([supply],
                                     columns=['i_ep', 'potato', 'tomato', 'cucumber', 'pakchoi', 'broccoli', 'cabbage',
                                              'turnip', 'lettuce', 'chinese_watermelon', 'green_bean', 'green_pepper',
                                              'eggplant', 'celery'])
        if i_ep == 0:
            result.to_csv(filename_w, mode='w', index=False, encoding='gbk')  # header参数为None表示不显示表头
            result_supply.to_csv(filename_supply, mode='w', index=False, encoding='gbk')
        else:
            result.to_csv(filename_w, mode='a', index=False, encoding='gbk', header=False)  # header参数为None表示不显示表头
            result_supply.to_csv(filename_supply, mode='a', index=False, encoding='gbk', header=False)

        # save ma_rewards
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    #time_end = time.time()  # 记录结束时间
    time_end = time.perf_counter()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('总运行时间为',time_sum)
    return rewards, ma_rewards



##WHY? 测试为什么用env_new.py文件里的函数？
def eval(cfg, env, agent):
    #time_start = time.time()  # 记录开始时间
    time_start = time.perf_counter()# 记录开始时间
    print('开始测试!')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    rewards = []
    ma_rewards = []  # moving average rewards
    supply = [[] for i in range(env.num_crop + 1)]
    cropping_plan = [[] for i in range(env.num_farmer)]
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0  # reward per episode
        state = env.reset()
        done = False
        while True:
            for f in cfg.farmer_all:
                is_current_done = env.is_current_done(f.current_crop, f.plant_month)
                action = agent.predict(state, is_current_done)
                #WHY env_new.py-->    def step(self, farmer, action, is_current_done): # 只是在一个农户的基础上执行动作并改变环境
                next_state, reward, done, _ = env.step(f, action, is_current_done)
                # print(done)
                state = next_state
                ep_reward += reward
                if done:
                    break
            if i_ep == 0:
                supply_in_step = env.get_supply()
                for i in range(env.num_crop + 1):
                    supply[i].append(supply_in_step[i])
                for i in range(env.num_farmer):
                    cropping_plan[i].append(cfg.farmer_all[i].current_crop)
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep + 1}/{cfg.eval_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    # time_end = time.time()  # 记录结束时间
    time_end = time.perf_counter()# 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('运行时间为', time_sum)
    return rewards, ma_rewards, supply, cropping_plan

''''#WHY 从run_multi_seed 导入参数
print(sys.argv)
#由打印的结果可知，sys.argv[1:]是命令行传递的参数,sys.argv[0]是命令行运行的文件名
random_seed = sys.argv[1]
setup_seed(random_seed)'''


if __name__ == "__main__":
    #WHY 从run_multi_seed 导入参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", default=25, type=int)
    # args = parser.parse_args()
    # random_seed = args.seed
    # print(random_seed)
    # setup_seed(random_seed)

    #单次运行
    # random_seed = Parameter.Random_Seed
    # setup_seed(random_seed)
    # if Parameter.ConsiderMiniSupply == True:
    #     Parameter.n_states = Parameter.num_crop * 4 + 1 + 4  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
    #
    # else:
    #     Parameter.n_states = Parameter.num_crop * 3 + 1 + 3  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
    # Parameter.n_actions = Parameter.num_crop + 1
    # writer = SummaryWriter()
    # cfg = DQNConfig()
    # # 训练
    # env, agent = env_agent_config(cfg, seed=Parameter.Random_Seed)
    # rewards, ma_rewards = train_Dynamic(cfg, env, agent, 0, 0)
    # make_dir(cfg.result_path, cfg.model_path)
    # agent.save(path=cfg.model_path)
    # save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    # plot_rewards(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    # plt.show()
    # writer.close()

    #多次运行
    begin_seed = 25
    end_seed = 126
    seed_lst = [x for x in range(begin_seed, end_seed)]
    for seed in seed_lst:
        random_seed = seed
        Parameter.Random_Seed = seed
        setup_seed(random_seed)
        if Parameter.ConsiderMiniSupply == True:
            Parameter.n_states = Parameter.num_crop * 4 + 1 + 4  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；

        else:
            Parameter.n_states = Parameter.num_crop * 3 + 1 + 3  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
        Parameter.n_actions = Parameter.num_crop + 1
        writer = SummaryWriter()
        cfg = DQNConfig()
        # 训练
        env, agent = env_agent_config(cfg, seed=Parameter.Random_Seed)
        rewards, ma_rewards = train_Dynamic(cfg, env, agent,0,0)
        make_dir(cfg.result_path, cfg.model_path)
        agent.save(path=cfg.model_path)
        save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
        plot_rewards(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
        writer.close()
    plt.show()

    '''
    env, agent = env_agent_config(cfg, seed=10)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards, supply, cropping_plan = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, tag="eval", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    plot_supply(supply, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    for i in cropping_plan:
        print(i)
    '''
    # 测试
