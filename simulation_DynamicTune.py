#WHY 不同合作社，不同策略的收益比较
import random
import numpy as np
from environment.farmer import farmer
from environment.env_v5_why_GBDTPrice import MyEnv
from demand_cal.action_pick import ActionPick
import simulation_plot
import Parameter
import train_for_Dynamic
from common.utils import save_results, make_dir
from common.plot import plot_rewards
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from price_data import decompose_GBDT_monthPredict
import pandas as pd
import csv
import os
from os.path import dirname, abspath
import copy
import matplotlib.patches as mpatches
import seaborn as sns
# from mpl_toolkits.axes_grid.axislines import SubplotZero

from collections import Counter
class CooperativePolicy:
    def __init__(self, id, UpdatedRL,UpdatedPredictPrice, cooperative_pro,NormalChoise,CosiderPrice,ConsiderMiniSupply,n_states,n_actions):
        self.id = id
        self.UpdatedRL = UpdatedRL
        self.UpdatedPredictPrice = UpdatedPredictPrice
        self.cooperative_pro = cooperative_pro
        self.NormalChoise = NormalChoise
        self.CosiderPrice = CosiderPrice
        self.ConsiderMiniSupply = ConsiderMiniSupply
        self.n_states = n_states
        self.n_actions = n_actions
    def ChangeCooperativeStatus(self):
        Parameter.UpdatedRL = self.UpdatedRL
        Parameter.UpdatedPredictPrice = self.UpdatedPredictPrice
        Parameter.cooperative_pro = self.cooperative_pro
        Parameter.NormalChoise = self.NormalChoise
        Parameter.CosiderPrice = self.CosiderPrice
        Parameter.ConsiderMiniSupply = self.ConsiderMiniSupply
        Parameter.n_states = self.n_states
        Parameter.n_actions = self.n_actions

    def Name(self):
        if self.UpdatedRL== False and self.NormalChoise == False:
            return str("Random Policy")
        elif self.UpdatedRL== False and self.NormalChoise == True and self.CosiderPrice == False:
            return str("Normal Policy")
        elif self.UpdatedRL== False and self.NormalChoise == True and self.CosiderPrice == True:
            return str("Normal Policy considering Price")
        elif self.UpdatedRL== True and self.ConsiderMiniSupply == False:
            return str("DQN without MiniSupply")
        elif self.UpdatedRL== True and self.ConsiderMiniSupply == True:
            return str("DQN considering MiniSupply")
        else:
            return str("Wrong")

def GetFileName(num_Cooperative,str_):
    os.makedirs('./outputs_dynamic/outputs', exist_ok=True)
    if Parameter.Tune_month_OR_Pre_month == 1:
        filename = './outputs_dynamic/outputs/Multi_Tune_month' + str(Parameter.Save_name)+'_Pre_month'+str(Parameter.Pre_month[0])+ '_cooperate' + str(num_Cooperative) + str(str_) + str(
            Parameter.Random_Seed) + '.csv'
    elif Parameter.Tune_month_OR_Pre_month == 2:
        filename = './outputs_dynamic/outputs/Multi_Pre_month' + str(Parameter.Save_name) + '_Tune_month'+ str(Parameter.Tune_month[0])+'_cooperate' + str(
                num_Cooperative) + str(str_) + str(
                Parameter.Random_Seed) + '.csv'
    elif Parameter.Tune_month_OR_Pre_month == 3:
        filename = './outputs_dynamic/outputs/Multi_Price' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) +'_Pre_month'+str(Parameter.Pre_month[0]) + '_cooperate' + str(
            num_Cooperative) + str(str_) + str(
            Parameter.Random_Seed) + '.csv'
    else:
        filename = './outputs_dynamic/outputs/Multi_cooperative' + str(Parameter.Save_name)+ '_Tune_month'+ str(Parameter.Tune_month[0])+'_Pre_month'+str(Parameter.Pre_month[0]) +'_cooperate' + str(
            num_Cooperative) +  str(str_) + str(
            Parameter.Random_Seed) + '.csv'
    return filename


def SaveProfitFile(data,num_month,num_Cooperative):

    filename = GetFileName(num_Cooperative, '_Profit_data_')
    #filename = './outputs_dynamic/' + str(num_Cooperative) + '_Profit_data_' + str(Parameter.Random_Seed) + '.csv'

    # WHY：第几个月，输出每个月的实际利润, 输出过去12个月的月均利润,输出截止到当月的总利润，轮更问题，产量为0
    # Parameter.Num_Month, profit_month, profit_12_month, profit_total, rotation_problem, plant_0_yeild

    data.insert(0, num_month)
    if Parameter.VirtualShow == 0 or Parameter.VirtualShow == 2 or Parameter.VirtualShow == 3:
        name = ['Num_Month', 'profit_month', 'profit_12past_month', 'profit_total', 'rotation_p',
         'plant_0_yeild', 'action_0']
        if Parameter.ConsiderMiniSupply == True:
            name.append('MiniSupply_problem')
            for crop in range(Parameter.num_crop):
                name.append(Parameter.action_name[crop+1]+'_notsatisify')
        result = pd.DataFrame(
            [data],
            columns=name)
    elif Parameter.VirtualShow == 1:
        name = ['Num_Month', 'profit_month', 'virtual_profit_month', 'profit_12past_month', 'virtual_profit_12past_month',
         'profit_total', 'virtual_profit_total', 'rotation_p',
         'plant_0_yeild', 'action_0']
        if Parameter.ConsiderMiniSupply == True:
            name.append('MiniSupply_problem')
            for crop in range(Parameter.num_crop):
                name.append(Parameter.action_name[crop+1]+'_notsatisify')
        result = pd.DataFrame(
            [data],
            columns=name)
    if num_month == 0:
        result.to_csv(filename, mode='w', index=False, encoding='gbk')  # header参数为None表示不显示表头
    else:
        result.to_csv(filename, mode='a', index=False, encoding='gbk', header=False)  # header参数为None表示不显示表头


def SaveSupplyFile(data,num_month, num_Cooperative):

    #filename = './outputs_dynamic/' + str(num_Cooperative) + '_Supply_data_' + str(Parameter.Random_Seed) + '.csv'
    filename = GetFileName(num_Cooperative, '_Supply_data_')
    data.insert(0, num_month)
    name = ['Num_Month']
    for i in range(Parameter.num_crop):
        name.append(Parameter.action_name[i + 1] + '_Total')
    if Parameter.ConsiderMiniSupply:  # 看最近12个月的作物总供给
        for i in range(Parameter.num_crop):
            # name.append(Parameter.action_name[i + 1] + '_Sum_Last12M')
            name.append(Parameter.action_name[i + 1] + '_LastYearSupply')

    result = pd.DataFrame([data], columns=name)
    if num_month == 0:
        result.to_csv(filename, mode='w', index=False, encoding='gbk')  # header参数为None表示不显示表头
    else:
        result.to_csv(filename, mode='a', index=False, encoding='gbk', header=False)  # header参数为None表示不显示表头



def SaveCropRotationFile(data,num_month,num_Cooperative):

    #filename = './outputs_dynamic/' + str(num_Cooperative) + '_CropRotation_data_' + str(Parameter.Random_Seed) + '.csv'
    filename = GetFileName(num_Cooperative, '_CropRotation_data_')
    columns_name = ['Num_Month']
    for i in range(Parameter.num_farmer):
        columns_name.append('F'+ str(i))
    xx = [num_month]
    for i in range(len(data)):
        xx.append(data[i])
    result = pd.DataFrame([xx], columns=columns_name)
    if num_month == 0:
        result.to_csv(filename, mode='w', index=False, encoding='gbk')  # header参数为None表示不显示表头
    else:
        result.to_csv(filename, mode='a', index=False, encoding='gbk', header=False)  # header参数为None表示不显示表头
def SaveCurrentCropFile(data,num_month,num_Cooperative):

    #filename = './outputs_dynamic/' + str(num_Cooperative) + '_CropRotation_data_' + str(Parameter.Random_Seed) + '.csv'
    filename = GetFileName(num_Cooperative, '_CurrentCrop_data_')
    columns_name = ['Num_Month']
    for i in range(Parameter.num_farmer):
        columns_name.append('F'+ str(i))
    xx = [num_month]
    for i in range(len(data)):
        xx.append(data[i])
    result = pd.DataFrame([xx], columns=columns_name)
    if num_month == 0:
        result.to_csv(filename, mode='w', index=False, encoding='gbk')  # header参数为None表示不显示表头
    else:
        result.to_csv(filename, mode='a', index=False, encoding='gbk', header=False)  # header参数为None表示不显示表头

def Normal_Farmer_Pick(current_month_,last_current_crop_,current_price, cost, min_demand, LastYearSupply):#考虑上个月价格
    new_action = ['null', 'potato', 'tomato', 'cucumber', 'pakchoi', 'broccoli', 'cabbage', 'turnip', 'lettuce',
                   'wax_gourd', 'bean', 'pepper', 'eggplant', 'celery']
    not_choose = []
    new_action_res = []
    profit = []
    current_price_res = [val for val in current_price]
    cost_price_res = [val for val in cost]
    cost_price_res.insert(0,0)
    # 前后顺序种植相同植物family作物
    if (last_current_crop_ in [1, 2, 11, 12]) :
        new_action = [n for i, n in enumerate(new_action) if i not in [1, 2, 11, 12]]

    elif (last_current_crop_ in [4, 5, 6, 7]) :
        new_action = [n for i, n in enumerate(new_action) if i not in [4, 5, 6, 7]]

    elif (last_current_crop_ in [3, 9]) :
        new_action = [n for i, n in enumerate(new_action) if i not in [3, 9]]

    for act in new_action:
        m = Parameter.action_name.index(act)
        if sum(Parameter.yeild[m][current_month_]) == -1:  # WHY:或该作物在当前月份种植时不能正常生长（产量 0）时
            not_choose.append(act)

    for act in not_choose:
        new_action.remove(act)



    if Parameter.ConsiderMiniSupply == True:#考虑最小供给
        for l in range(len(new_action)):
            new_action_res.append(new_action[l])
        for crop in range(1, Parameter.num_crop + 1):
            m_d = 1.0 * min_demand[crop] - sum(LastYearSupply[crop])
            # m_d = 1.0 * min_demand[crop]
            if m_d <= 0 and Parameter.action_name[crop] in new_action:
                new_action.remove(Parameter.action_name[crop])
        # if len(new_action) == 0 != len(new_action_res):
        print('new_action is :',  new_action)
        print('new_action_res is :',  new_action_res)
        if len(new_action) == 0 and len(new_action_res)!= 0:
            for l in range(len(new_action_res)):
                new_action.append(new_action_res[l])


    if Parameter.CosiderPrice == True:#考虑上个月价格因素
        if Parameter.ConsiderMiniSupply == True and len(new_action) != len(new_action_res): #如果有最小供给的作物要种，要考虑从这里面随机选择，先不考虑价格的问题了
            if len(new_action) != 0:
                index = random.randint(0, len(new_action) - 1)
                n_act = new_action[index]
                action = Parameter.action_name.index(n_act)
        else:
            if len(new_action)!=0:
                current_price_last =[current_price_res[Parameter.vegetable.index(act) + 1] for act in new_action]
                cost_last = [cost_price_res[Parameter.vegetable.index(act) + 1] for act in new_action]
                #index_max = [i for i, val in enumerate(current_price_last) if val == max(current_price_last)]
                harvest_time = [Parameter.yeild[Parameter.vegetable.index(act) + 1][current_month_ % 12].index(-1) - 1 for act in new_action]
                inteval_month = [(time - current_month_) for time in harvest_time]


                for crop in range(len(current_price_last)):
                    profit.append(current_price_last[crop] - cost_last[crop])  # current_price、supply14位，price、plant_cost13位
                    assert (inteval_month[crop] != 0)
                    profit[crop] /= inteval_month[crop]
                index_max = [i for i, val in enumerate(profit) if val == max(profit)]

                index = random.sample(index_max, 1)#如果有多个一样的就随机选一个
                n_act = new_action[index[0]]
                action = Parameter.action_name.index(n_act)

                # if n_act == 'lettuce' or n_act == 'bean':
                print('vegetable is %s, price is %d, cost is %d' %(n_act,current_price_last[index[0]], cost_last[index[0]]))
            else:

                for crop in range(1, len(current_price_res)):
                    profit.append(
                        current_price_res[crop] - cost_price_res[crop])  # current_price、supply14位，price、plant_cost13位,cost_price_res插入1位变为14位了

                    harvest_time = Parameter.yeild[crop][current_month_ % 12].index(-1) - 1
                    inteval_month = harvest_time - current_month_
                    assert (inteval_month != 0)
                    profit[crop] /= inteval_month

                index_max = [i for i, val in enumerate(profit) if val == max(profit)]
                index = random.sample(index_max, 1)
                action = Parameter.action_name[index[0]]


    else:
        if len(new_action)!=0:
            index =random.randint(0,len(new_action)-1)
            n_act = new_action[index]
            action = Parameter.action_name.index(n_act)
        else:

            action = Parameter.GlobalRand[Parameter.GlobalRand_index]
            Parameter.GlobalRand_index += 1

    if action !=0:
        if cost[action - 1] > current_price[action]:##current_price14位，price、plant_cost13位
            print('cost is larger than price,current month is %s, choose action is %d,price is %f, cost is %f' %(current_month_, action,current_price[action],cost[action - 1]))

    return action

def VritualCauProfit(env,current_month,rotation, num_month):
    profit_month = {}
    # WHY:获取未来月份所有农户的每个月总供给量
    supply = [[0 for i in range(8)] for i in range(Parameter.num_crop + 1)]

    future_month = []
    for i in range(len(env.farmer_all)):
        f = env.farmer_all[i]
        harvest_time = Parameter.yeild[f.current_crop][f.plant_month % 12].index(-1) - 1
        inteval_month = harvest_time - current_month % 12
        if inteval_month < 0:
            inteval_month += 12

        future_month.append(inteval_month)
        for j in range (inteval_month):
            assert (Parameter.yeild[f.current_crop][f.plant_month][(current_month + j + 1) % 12] != -1)
            supply[f.current_crop][j] += f.area * Parameter.yeild[f.current_crop][f.plant_month][(current_month + j + 1) % 12]

    future_month = list(set(future_month))#去除重复
    future_month.sort()#从小到大排序

    # 计算合作社的月收益——用于计算产生reward
    if future_month[0] < 0 or future_month[-1] > 8:
        print('生育期小于或大于正常值')
    if future_month[0] == 0:#空动作
        l = 1
    elif future_month[0] >= num_month:#超过了更新算法的周期的话，采用算法周期值
        l = num_month - 1
    else:
        l = future_month[0]
    for m in range (l):
        profit = 0
        for i in range(1, (Parameter.num_crop + 1)):
           profit += supply[i][m] * (env.get_price(i - 1 ,m+current_month+1) - env.get_cost(i - 1))

        profit = profit * (0.8 ** rotation)
        profit_month[m+current_month+1]  = profit

    return  profit_month
def Update_Price(env,num_month,num_cooperate):#每隔num_month动态更新RL算法
    if Parameter.Tune_month_OR_Pre_month != 3:  # 价格没变化
        num_cooperate = 0
    if Parameter.UpdatedPredictPrice == False:#不动态更新价格预测算法，根据确定好的算法更新价格列表
        if Parameter.VirtualShow == 0:
            env.Update_PriceList(Parameter.Num_Month + num_month,
                                 0,num_cooperate)  # 需要Parameter.Num_Month+num_month个月的真实数据，预测num_month个月价格准备好
        elif Parameter.VirtualShow == 1 or Parameter.VirtualShow == 2 or Parameter.VirtualShow == 3:
            env.Update_PriceList(Parameter.Num_Month, num_month + 8, num_cooperate)  # 预测num_month或8个月数据

    else:
        if Parameter.VirtualShow == 0 :
            env.MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month + num_month, num_month,num_cooperate,False) #需要从第Parameter.Num_Month个月开始预测
        elif Parameter.VirtualShow == 1 or Parameter.VirtualShow == 2 or Parameter.VirtualShow == 3:
            env.MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month,num_month,num_cooperate,False)# 预测num_month或8个月数据

def FarmerState(env,obs,f):
    state = []
    state.extend(obs)
    state.append(env.farmer_all[f].area * 2 - 3)
    state.append(env.farmer_all[f].current_crop * 2 / 13 - 1)
    state.append(env.farmer_all[f].plant_month * 2 / 11 - 1)
    if Parameter.ConsiderMiniSupply == True:
        f_total_yeild = sum(
            Parameter.yeild[env.farmer_all[f].current_crop][env.farmer_all[f].plant_month % 12]) + 1
        s4 = f_total_yeild * 0.5 - 1.25  # 产量
        state.append(s4)

    if Parameter.ConsiderMiniSupply == True:  # 考虑最小供给，把剩余的需求放入状态

        # 整个合作社的最小供给情况
        Obs = []
        for crop in range(1, Parameter.num_crop + 1):
            if env.min_demand[crop] <= 0:
                Obs.append(1)
            else:
                # if env.min_demand[crop] > 0:
                if Parameter.YearOrMonthSupply == False:  # 如果是稳定按月供给，这块没仔细考虑！！！
                    m_d = 1 * (env.min_demand[crop] - sum(env.supply_in_last_12[crop])) / Parameter.num_farmer
                    last_demand = m_d - sum(env.farmer_all[f].supply_in_last_12[crop])
                    if last_demand >= 0:
                        Obs.append((1 - 2 * last_demand / m_d))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    else:  # 需求满足
                        Obs.append(1)
                elif Parameter.YearOrMonthSupply == True:  # 如果是按照年供给
                    assert (len(env.LastYearSupply[crop]) <= 12)

                    m_d = 1 * (env.min_demand[crop] - sum(env.LastYearSupply[crop]))
                    if m_d >= 0:
                        assert (env.min_demand[crop] > 0)
                        Obs.append((1 - 2 * m_d / env.min_demand[crop]))  # 最小需求是否满足 1-2x  -> -1 ~ 1
                    else:  # 需求满足
                        Obs.append(1)
        state.extend(Obs)

    return state
def CalVirtualProfit(env,rl_p, num_month,obs,model_path, profit_lasttotal,supply_lasttotal,num_cooperate):
    profit_month = []  # 输出每个月的实际利润
    profit_12_month = []  # 输出过去12个月的月均利润
    profit_total = []  # 输出截止到当月的总利润
    virtual_profit_total = []  # 输出截止到当月的虚拟总利润
    virtual_profit_month = []  # 输出每个月的虚拟利润
    virtual_profit_12_month = []  # 输出过去12个月的月均利润
    supply_total = [[] for i in range(Parameter.num_crop)]  # 输出最近一个月的总供给
    Sum_supply_in_last_12 = [0.0 for i in range(Parameter.num_crop + 1)]  # 最近12个月的总供给
    rotation_problem_test = []
    plant_0_yeild_test = []
    action_0 = []
    flag = 0
    env_res = copy.deepcopy(env)#深拷贝,形成2个不同对象
    Last_month = Parameter.Max_month - env_res.step_num - 1
    start_month = env_res.step_num
    # Last_month = Parameter.Max_month - Parameter.Num_Month - 1

    # 更新价格
    Update_Price(env_res, Last_month + num_month, num_cooperate)

    if Last_month > 0:
        for e in range(Last_month):
            action = []
            action_Each_Month = []
            crop_Each_Month = []
            for f in range(Parameter.num_farmer):
                state = FarmerState(env_res,obs,f)
                if env_res.is_current_done(env_res.farmer_all[f].current_crop, env_res.farmer_all[f].plant_month):
                    if random.random() < rl_p:
                        if Parameter.UpdatedRL:  # 选择每次更新的RL算法
                            action_f = actionPick.rl_pick(state, Parameter.n_states, Parameter.n_actions, model_path)
                        else:
                            action_f = actionPick.rl_pick_static(state, Parameter.n_states, Parameter.n_actions, model_path)
                        action.append(action_f)
                        env_res.farmer_all[f].FarmerShedule(e + Parameter.Num_Month, action_f)
                    else:
                        if Parameter.NormalChoise == False:
                            # action_f = random.randint(1, 13) #用局部的randint生成的是非均匀的随机数分布，奇怪，改成下面的全局随机数了
                            action_f = Parameter.GlobalRand[Parameter.GlobalRand_index]
                            Parameter.GlobalRand_index += 1
                        else:
                            env_res._set_current_price(e + Parameter.Num_Month)  # 考虑这个月价格
                            action_f = Normal_Farmer_Pick((e + Parameter.Num_Month) % 12, env_res.farmer_all[f].current_crop,
                                                          env_res.get_current_price(), env_res.plant_cost, env_res.min_demand,
                                                          env_res.LastYearSupply)
                        action.append(action_f)
                        env_res.farmer_all[f].FarmerShedule(e + Parameter.Num_Month, action_f)
                else:
                    action.append(0)
                    # env.farmer_all[f].FarmerShedule(e+Parameter.Num_Month - num_month, 0)# 此时，已经Num_Month += num_month,但是还没达到
                    env_res.farmer_all[f].FarmerShedule(e + Parameter.Num_Month, 0)
                # 保存行动数据
                # action_Each_Month.append(env.farmer_all[f].GetActionShedule(e+Parameter.Num_Month - num_month))# 此时，已经Num_Month += num_month,但是还没达到
                action_Each_Month.append(env_res.farmer_all[f].GetActionShedule(e + Parameter.Num_Month))
                crop_Each_Month.append(env_res.farmer_all[f].current_crop)
            obs, reward, done, info = env_res.step(action)

            # 输出每个月的实际利润
            profit_month.append(round(info['actual_profit'], 2))
            if Parameter.VirtualShow == 1:
                if e == 0:
                    virtual_profit_month.insert(0, profit_month[-1])
            # 输出过去12个月的月均利润
            profit_12_month.append(round(info['profit'], 2))
            # 输出截止到当月的总利润
            if (len(profit_total) == 0):
                profit_total.append(round(info['actual_profit'], 2) + profit_lasttotal)

            else:
                profit_total.append(profit_total[-1] + round(info['actual_profit'], 2))

            for crop in range(Parameter.num_crop):
                if (len(supply_total[crop]) == 0):
                    supply_total[crop].append(round(info['supply'][crop + 1], 3) + supply_lasttotal[crop])  # WHY实际作物供给
                else:
                    supply_total[crop].append(
                        supply_total[crop][-1] + round(info['supply'][crop + 1], 3))  # WHY实际作物供给

            rotation_problem_test.append(info['rotation_problem'])
            plant_0_yeild_test.append(info['plant_0_yeild'])
            action_0.append(info['action_0'])

        VirtualTotalProfit[num_cooperate].update({(start_month): profit_total})

def ProbRL_Dynamic_in_next(env,rl_p, num_month,obs,model_path, profit_lasttotal,supply_lasttotal,num_cooperate):#以一定概率rl_p来选择RL,每隔num_month动态更新RL算法，

    profit_month = [] # 输出每个月的实际利润
    profit_12_month = []  # 输出过去12个月的月均利润
    profit_total = []  # 输出截止到当月的总利润
    virtual_profit_total = []  # 输出截止到当月的虚拟总利润
    virtual_profit_month = []  # 输出每个月的虚拟利润
    virtual_profit_12_month = []  # 输出过去12个月的月均利润
    supply_total = [[] for i in range(Parameter.num_crop)]  # 输出最近一个月的总供给
    Sum_supply_in_last_12 = [0.0 for i in range(Parameter.num_crop + 1)]  # 最近12个月的总供给
    rotation_problem_test = []
    plant_0_yeild_test = []
    action_0 = []
    flag = 0

    #更新价格
    Update_Price(env, num_month, num_cooperate)
    if Parameter.VirtualShow == 2:
        #VirtualMonthPrice[num_cooperate].append({Parameter.Num_Month:env.price})
        VirtualMonthPrice[num_cooperate].update({Parameter.Num_Month:env.price})#字典添加元素

    for e in range(num_month):
        action = []
        action_Each_Month = []
        crop_Each_Month = []
        for f in range(Parameter.num_farmer):
            state = FarmerState(env,obs,f)
            if env.is_current_done(env.farmer_all[f].current_crop, env.farmer_all[f].plant_month):
                if random.random() < rl_p:
                    if Parameter.UpdatedRL:#选择每次更新的RL算法
                        action_f = actionPick.rl_pick(state, Parameter.n_states, Parameter.n_actions,model_path)
                    else:
                        action_f = actionPick.rl_pick_static(state, Parameter.n_states, Parameter.n_actions, model_path)
                    action.append(action_f)
                    #env.farmer_all[f].FarmerShedule(e+Parameter.Num_Month - num_month, action_f)# 此时，已经Num_Month += num_month,但是还没达到
                    env.farmer_all[f].FarmerShedule(e + Parameter.Num_Month, action_f)
                else:
                    if Parameter.NormalChoise == False:
                        # action_f = random.randint(1, 13) #用局部的randint生成的是非均匀的随机数分布，奇怪，改成下面的全局随机数了
                        action_f = Parameter.GlobalRand[Parameter.GlobalRand_index]
                        Parameter.GlobalRand_index += 1
                    else:
                        env._set_current_price(e + Parameter.Num_Month)#考虑这个月价格
                        action_f = Normal_Farmer_Pick((e + Parameter.Num_Month)%12,env.farmer_all[f].current_crop ,env.get_current_price(),env.plant_cost,env.min_demand, env.LastYearSupply)
                    action.append(action_f)
                    #env.farmer_all[f].FarmerShedule(e+Parameter.Num_Month - num_month, action_f)# 此时，已经Num_Month += num_month,但是还没达到
                    env.farmer_all[f].FarmerShedule(e + Parameter.Num_Month, action_f)
            else:
                action.append(0)
                #env.farmer_all[f].FarmerShedule(e+Parameter.Num_Month - num_month, 0)# 此时，已经Num_Month += num_month,但是还没达到
                env.farmer_all[f].FarmerShedule(e + Parameter.Num_Month, 0)
            # 保存行动数据
            #action_Each_Month.append(env.farmer_all[f].GetActionShedule(e+Parameter.Num_Month - num_month))# 此时，已经Num_Month += num_month,但是还没达到
            action_Each_Month.append(env.farmer_all[f].GetActionShedule(e + Parameter.Num_Month))
            crop_Each_Month.append(env.farmer_all[f].current_crop)
        obs, reward, done, info = env.step(action)

        #计算虚拟的价格
        if Parameter.VirtualShow == 1:
            profit_total_next_month = 0
            if flag == e:
                virtual_profit_month_ = VritualCauProfit(env, Parameter.Num_Month + e, info['rotation_problem'],num_month)
                l = len(virtual_profit_month_)
                if num_month - len(virtual_profit_month) < l:
                    for i in range(num_month - len(virtual_profit_month)):
                        profit_total_next_month += virtual_profit_month_[Parameter.Num_Month + e + i + 1]
                        virtual_profit_month.append(profit_total_next_month)
                    flag += num_month - e
                else:
                    for key in virtual_profit_month_:
                        profit_total_next_month += virtual_profit_month_[key]
                        virtual_profit_month.append(profit_total_next_month)
                    flag += l

        # 输出每个月的实际利润
        profit_month.append(round(info['actual_profit'], 2))
        if Parameter.VirtualShow == 1:
            if e == 0:
                virtual_profit_month.insert(0, profit_month[-1])
        # 输出过去12个月的月均利润
        profit_12_month.append(round(info['profit'], 2))
        # 输出截止到当月的总利润
        if (len(profit_total) == 0):
            profit_total.append(round(info['actual_profit'], 2) + profit_lasttotal)

        else:
            profit_total.append(profit_total[-1] + round(info['actual_profit'], 2))

        for crop in range(Parameter.num_crop):
            if (len(supply_total[crop]) == 0):
                supply_total[crop].append(round(info['supply'][crop + 1], 3) + supply_lasttotal[crop])  # WHY实际作物供给
            else:
                supply_total[crop].append(
                    supply_total[crop][-1] + round(info['supply'][crop + 1], 3))  # WHY实际作物供给

        rotation_problem_test.append(info['rotation_problem'])
        plant_0_yeild_test.append(info['plant_0_yeild'])
        action_0.append(info['action_0'])

        # 保存数据
        if Parameter.VirtualShow == 0 or Parameter.VirtualShow == 2 or Parameter.VirtualShow == 3:
            data = [profit_month[-1], profit_12_month[-1], profit_total[-1],
             round(info['rotation_problem'], 2), round(info['plant_0_yeild'], 2), round(info['action_0'], 2)]
            if Parameter.ConsiderMiniSupply == True:
                data.append(round(info['MiniSupply_problem'], 2))
                for crop in range(Parameter.num_crop):
                    data.append(info['MiniSupply_problem_list'][crop+1])

            SaveProfitFile(data,
                       e + Parameter.Num_Month, num_cooperate)  # 此时，已经Num_Month += num_month,但是还没达到
        elif Parameter.VirtualShow == 1:
            total = sum(virtual_profit_month[:(e+1)]) + profit_lasttotal
            if (e < 12):
                month_12 = profit_12_month[e] + (sum(virtual_profit_month[:e+1]) - sum(profit_month[:e+1])) / 12
            else:
                month_12 = profit_12_month[e] + (sum(virtual_profit_month[e-11: e+1]) - sum(profit_month[e-11:e+1])) / 12
            data = [profit_month[-1], virtual_profit_month[e], profit_12_month[-1], month_12, profit_total[-1], total,
             round(info['rotation_problem'], 2), round(info['plant_0_yeild'], 2),
             round(info['action_0'], 2)]
            if Parameter.ConsiderMiniSupply == True:
                data.append(round(info['MiniSupply_problem'], 2))
                for crop in range(Parameter.num_crop):
                    data.append(info['MiniSupply_problem_list'][crop+1])

            SaveProfitFile(data,
                           e + Parameter.Num_Month, num_cooperate)  # 此时，已经Num_Month += num_month,但是还没达到
        xx = []
        for crop in range(Parameter.num_crop):
            xx.append(supply_total[crop][-1])
        if Parameter.ConsiderMiniSupply:  # 看最近12个月的作物总供给
            for crop in range(Parameter.num_crop + 1):
                # Sum_supply_in_last_12[crop] = sum(info['supply_in_last_12'][crop])
                Sum_supply_in_last_12[crop] = sum(info['LastYearSupply'][crop])
            for i in range(Parameter.num_crop):
                xx.append(Sum_supply_in_last_12[i + 1])
        SaveSupplyFile(xx, e + Parameter.Num_Month, num_cooperate)
        SaveCropRotationFile(action_Each_Month, e + Parameter.Num_Month, num_cooperate)
        SaveCurrentCropFile(crop_Each_Month, e + Parameter.Num_Month, num_cooperate)
        Crop_total_bar[num_cooperate].append(action_Each_Month)

    if Parameter.VirtualShow == 3:
        CalVirtualProfit(env,rl_p, num_month,obs,model_path, profit_total[-1],supply_lasttotal,num_cooperate)

    print('所有农户以', rl_p * 100, '%的概率根据RL选择，否则随机选：')

    for i in range (len(virtual_profit_month)):
        virtual_profit_total.append(sum(virtual_profit_month[:(i + 1)]) + profit_lasttotal)
        if ( i < 12):
            virtual_profit_12_month.append(profit_12_month[i] + (sum(virtual_profit_month[:i+1]) - sum(profit_month[:i+1])) / 12)
        else:
            virtual_profit_12_month.append(profit_12_month[i] + (sum(virtual_profit_month[i-11: i+1]) - sum(profit_month[i-11: i+1])) / 12)
    return env, profit_12_month, virtual_profit_12_month, profit_month,virtual_profit_month, profit_total,virtual_profit_total,supply_total


def cooperative_DynamicCmpare(cooperative_env,cooperative_pro,cooperative_obs,Tune_month, pre_month):#本函数用训练一次的算法，但是用不同的概率用于不同合作社
    #cooperative_pro 中合作概率为0表示随机，cooperative_pro为1表示完全采用RL
    print("第" + str(Parameter.Num_Month) + "个月开始")
    print('最开始农户随机选择：')
    #Parameter.Num_Month +=  pre_month  #初始的Pre_month月各合作社是一样的策略和动作,Pre_month个月用来预训练。全局变量已经到了第num_month月

    profit_12_month = [[] for i in range(len(cooperative_env))] # 输出过去12个月的月均利润
    profit_month = [[] for i in range(len(cooperative_env))]  # 输出每个月的实际利润
    profit_total =[[] for i in range(len(cooperative_env))]  # 输出截止到当月的总利润
    virtual_profit_total = [[] for i in range(len(cooperative_env))]  # 输出截止到当月的虚拟总利润
    virtual_profit_month = [[] for i in range(len(cooperative_env))]  # 输出每个月的虚拟利润
    virtual_profit_12_month = [[] for i in range(len(cooperative_env))]  # 输出过去12个月的月均虚拟利润
    supply_total = [[[] for i in range(Parameter.num_crop)] for i in range(len(cooperative_env))]  # 输出最近一个月的总供给
    Sum_supply_in_last_12 = [[0.0 for i in range(Parameter.num_crop + 1)] for i in
                             range(len(cooperative_env))]  # 最近12个月的总供给

    #最初一样
    action = []
    #更新价格
    for i in range(len(cooperative_env)):
        if Parameter.UpdatedPredictPrice == False:  # 不动态更新价格预测算法，根据确定好的算法更新价格列表
            if Parameter.Tune_month_OR_Pre_month == 3:
                cooperative_env[i].Update_PriceList(Parameter.Num_Month + pre_month, 0 , i)  # 价格准备好
                if i == 0:
                    label_.append('price initial ' )
                else:
                    label_.append('price strategy ' + str (i))
            else:
                cooperative_env[i].Update_PriceList(Parameter.Num_Month + pre_month, 0, 0)  # 价格准备好
                label_.append('%' + str(cooperative_pro[i]) + ' rl model')

        else:
            if Parameter.Tune_month_OR_Pre_month == 3:
                cooperative_env[i].MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month + pre_month, 0,
                                            i,False)  # 需要从第Parameter.Num_Month个月开始预测
                if i == 0:
                    label_.append('price initial ')
                else:
                    label_.append('price strategy ' + str(i))
            else:
                cooperative_env[i].MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month + pre_month, 0, 0, False)  # 价格准备好
                label_.append('%' + str(cooperative_pro[i]) + ' rl model')

        if Parameter.VirtualShow == 2:
            #VirtualMonthPrice[i].append({Parameter.Num_Month: cooperative_env[i].price})
            VirtualMonthPrice[i].update({Parameter.Num_Month: cooperative_env[i].price})



    for j in range(pre_month):
        action_Each_Month = []
        crop_Each_Month = []
        for i in range(len(cooperative_env)):
            if i == 0:
                for f in range(Parameter.num_farmer):
                    # 随机选择动作
                    if cooperative_env[0].is_current_done(cooperative_env[0].farmer_all[f].current_crop, cooperative_env[0].farmer_all[f].plant_month):
                        if Parameter.NormalChoise == False:
                            # action_f = random.randint(1, 13)
                            action_f = Parameter.GlobalRand[Parameter.GlobalRand_index]
                            Parameter.GlobalRand_index += 1
                        else:

                            action_f = Parameter.GlobalRand[Parameter.GlobalRand_index]
                            Parameter.GlobalRand_index += 1

                        cooperative_env[0].farmer_all[f].current_crop = action_f
                        cooperative_env[0].farmer_all[f].plant_month = (j) % 12
                        action.append(action_f)
                        cooperative_env[0].farmer_all[f].FarmerShedule(j, action_f)
                    else:
                        action.append(0)
                        cooperative_env[0].farmer_all[f].FarmerShedule(j, 0)

                    #保存行动数据
                    action_Each_Month.append(cooperative_env[0].farmer_all[f].GetActionShedule(j))
                    crop_Each_Month.append(cooperative_env[0].farmer_all[f].current_crop)

                obs, reward, done, info = cooperative_env[0].step(action)

                # 输出过去12个月的月均利润
                profit_12_month[i].append(round(info['profit'], 2))
                # 输出每个月的实际利润
                profit_month[i].append(round(info['actual_profit'], 2))

                if Parameter.VirtualShow == 1:
                    virtual_profit_12_month[i].append(round(info['profit'], 2))
                    virtual_profit_month[i].append(round(info['actual_profit'], 2))
                # 输出截止到当月的总利润
                if (len(profit_total[i]) == 0) :
                    profit_total[i].append(round(info['actual_profit'], 2))
                    if Parameter.VirtualShow == 1:
                        virtual_profit_total[i].append(round(info['actual_profit'], 2))
                else:
                    profit_total[i].append(profit_total[i][-1] + round(info['actual_profit'], 2))
                    if Parameter.VirtualShow == 1:
                        virtual_profit_total[i].append(profit_total[i][-1] + round(info['actual_profit'], 2))

                for crop in range(Parameter.num_crop):
                    if (len(supply_total[i][crop]) == 0):
                        supply_total[i][crop].append(round(info['supply'][crop + 1], 3))# WHY实际作物供给
                    else:
                        supply_total[i][crop].append(supply_total[i][crop][-1] + round(info['supply'][crop + 1], 3))# WHY实际作物供给
                for crop in range(Parameter.num_crop + 1):
                    # Sum_supply_in_last_12[i][crop] = round(sum(info['supply_in_last_12'][crop]), 3)  # 看最近12个月的作物总供给
                    Sum_supply_in_last_12[i][crop] = round(sum(info['LastYearSupply'][crop]), 3)  # 看最近12个月的作物总供给
            else:
                assert (len(action) != 0)
                cooperative_env[i].step(action[-Parameter.num_farmer:])
                for f in range(Parameter.num_farmer):
                    cooperative_env[i].farmer_all[f].FarmerShedule(j, action[-Parameter.num_farmer:][f])
                    # 保存行动数据
                    # action_Each_Month.append(cooperative_env[0].farmer_all[f].GetActionShedule(j))
                # 输出过去12个月的月均利润
                profit_12_month[i].append(profit_12_month[0][j])
                # 输出每个月的实际利润
                profit_month[i].append(profit_month[0][j])
                # 输出截止到当月的总利润
                profit_total[i].append(profit_total[0][j])

                if Parameter.VirtualShow == 1:
                    virtual_profit_12_month[i].append(profit_12_month[0][j])
                    virtual_profit_month[i].append(profit_month[0][j])
                    virtual_profit_total[i].append(profit_total[0][j])

                # 输出截止到当月的总供给
                for crop in range(Parameter.num_crop):
                    supply_total[i][crop].append(supply_total[0][crop][j])# WHY实际作物供给

                for crop in range(Parameter.num_crop + 1):
                    Sum_supply_in_last_12[i][crop] = (Sum_supply_in_last_12[0][crop])  # 看最近12个月的作物总供给

            # 保存数据
            if Parameter.VirtualShow == 0 or Parameter.VirtualShow == 2 or Parameter.VirtualShow == 3:
                data = [profit_month[i][-1], profit_12_month[i][-1], profit_total[i][-1],
                 round(info['rotation_problem'], 2), round(info['plant_0_yeild'], 2), round(info['action_0'], 2)]
                if Parameter.ConsiderMiniSupply == True:
                    data.append(round(info['MiniSupply_problem'], 2))
                    for crop in range(Parameter.num_crop):
                        data.append(info['MiniSupply_problem_list'][crop + 1])
                SaveProfitFile(data, j, i)
            elif Parameter.VirtualShow == 1:
                data = [profit_month[i][-1],virtual_profit_month[i][-1], profit_12_month[i][-1],virtual_profit_12_month[i][-1], profit_total[i][-1],virtual_profit_total[i][-1],
                            round(info['rotation_problem'], 2), round(info['plant_0_yeild'], 2),round(info['action_0'], 2) ]
                if Parameter.ConsiderMiniSupply == True:
                    data.append(round(info['MiniSupply_problem'], 2))
                    for crop in range(Parameter.num_crop):
                        data.append(info['MiniSupply_problem_list'][crop + 1])
                SaveProfitFile(data, j, i)
            xx = []
            for crop in range(Parameter.num_crop):
                xx.append(supply_total[i][crop][-1])

            if Parameter.ConsiderMiniSupply:  # 看最近12个月的作物总供给
                for crop in range(Parameter.num_crop):
                    xx.append(Sum_supply_in_last_12[i][crop + 1])

            SaveSupplyFile(xx, j, i)
            Crop_total_bar[i].append(action_Each_Month)
            SaveCropRotationFile(action_Each_Month, j, i)
            SaveCurrentCropFile(crop_Each_Month, j, i)

    profit_12_plot.append(profit_12_month)
    profit_month_plot.append(profit_month)
    profit_total_plot.append(profit_total)
    if Parameter.VirtualShow == 1:
        virtual_profit_month_plot.append(virtual_profit_month)
        virtual_profit_total_plot.append(virtual_profit_total)
        virtual_profit_12_plot.append(virtual_profit_12_month)
    supply_total_plot.append(supply_total)
    x_1 = [i for i in range(1, pre_month + 1)]
    x.append(x_1)
    Parameter.Num_Month += pre_month  # 初始的Pre_month月各合作社是一样的策略和动作,Pre_month个月用来预训练。全局变量已经到了第num_month月
    print("运行至第" + str(Parameter.Num_Month) + "个月")
    #开始不同合作社的比较了，或不同参数曲线的比较
    while (True):

        if cooperative_env[0].step_num + Tune_month >= Parameter.Max_month:
            num_month = Parameter.Max_month - cooperative_env[0].step_num - 1
        else:
            num_month = Tune_month

        profit_12_month = [[] for i in range(len(cooperative_env))]  # 输出过去12个月的月均利润
        profit_month = [[] for i in range(len(cooperative_env))]  # 输出每个月的实际利润
        profit_total = [[] for i in range(len(cooperative_env))]  # 输出截止到当月的总利润
        virtual_profit_month = [[] for i in range(len(cooperative_env))]  # 输出截止到当月的虚拟总利润
        virtual_profit_total = [[] for i in range(len(cooperative_env))]  # 输出截止到当月的虚拟总利润
        virtual_profit_12_month = [[] for i in range(len(cooperative_env))]  # 输出过去12个月的月均虚拟利润
        supply_total = [[[] for i in range(Parameter.num_crop)] for i in range(len(cooperative_env))]  # 输出最近一个月的总供给
        #更新RL算法
        random_seed = Parameter.Random_Seed
        train_for_Dynamic.setup_seed(random_seed)
        cfg = train_for_Dynamic.DQNConfig()
        if Parameter.UpdatedRL and Parameter.Tune_month_OR_Pre_month !=3:
            print("第" + str(Parameter.Num_Month) + "个月开始更新RL算法")

            if Parameter.SelfAdaption_Step_or_data == True:
                if Parameter.Num_Month >= 70:
                    Parameter.Step_or_data = 1
                    Parameter.Train_Eps = 200  # 训练的回合数
                    Parameter.Max_Step = 120  # 每回合迭代最大步长
                else:
                    Parameter.Step_or_data = 0
                    Parameter.Train_Eps = 150  # 训练的回合数
                    Parameter.Max_Step = 120  # 每回合迭代最大步长

            writer = SummaryWriter()
            # 训练
            env, agent = train_for_Dynamic.env_agent_config(cfg, seed=Parameter.Random_Seed)
            rewards, ma_rewards = train_for_Dynamic.train_Dynamic(cfg, env, agent,num_month,0)
            make_dir(cfg.result_path, cfg.model_path)
            agent.save(path=cfg.model_path)
            save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
            # plot_rewards(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
            writer.close()

        print("第" + str(Parameter.Num_Month) + "个月,开始连续" + str(num_month) + "个月运行RL 算法")
        # 连续num_month个月运行RL 算法
        for i in range(len(cooperative_env)):

            if Parameter.UpdatedRL and Parameter.Tune_month_OR_Pre_month == 3:#如果比较不同价格就需要不同的算法
                Parameter.Save_name = Parameter.Price_Change[i]

                # 更新RL算法
                random_seed = Parameter.Random_Seed
                train_for_Dynamic.setup_seed(random_seed)
                cfg = train_for_Dynamic.DQNConfig()
                print("第" + str(Parameter.Num_Month) + "个月开始更新RL算法")

                if Parameter.SelfAdaption_Step_or_data == True:
                    if Parameter.Num_Month >= 70:
                        Parameter.Step_or_data = 1
                        Parameter.Train_Eps = 200  # 训练的回合数
                        Parameter.Max_Step = 120  # 每回合迭代最大步长
                    else:
                        Parameter.Step_or_data = 0
                        Parameter.Train_Eps = 150  # 训练的回合数
                        Parameter.Max_Step = 120  # 每回合迭代最大步长

                writer = SummaryWriter()
                # 训练
                env, agent = train_for_Dynamic.env_agent_config(cfg, seed=Parameter.Random_Seed)
                rewards, ma_rewards = train_for_Dynamic.train_Dynamic(cfg, env, agent,num_month, i)
                make_dir(cfg.result_path, cfg.model_path)
                agent.save(path=cfg.model_path)
                save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
                # plot_rewards(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
                writer.close()

            supply_lasttotal = []
            for crop in range(Parameter.num_crop):
                supply_lasttotal.append(supply_total_plot[-1][i][crop][-1])
            #cooperative_env[i].set_current_month(Parameter.Num_Month)
            cooperative_env[i], profit_12_month[i], virtual_profit_12_month[i], profit_month[i], virtual_profit_month[
                i], profit_total[i], virtual_profit_total[i], supply_total[i] = ProbRL_Dynamic_in_next(cooperative_env[i],
                                                                                               cooperative_pro[i],
                                                                                               num_month,
                                                                                               cooperative_env[
                                                                                                   i]._get_observation(),
                                                                                               cfg.model_path,
                                                                                               profit_total_plot[-1][i][
                                                                                                   -1],
                                                                                               supply_lasttotal, i)


            profit_12_month[i].insert(0, profit_12_plot[-1][i][-1])# 输出过去12个月的月均利润
            profit_month[i].insert(0, profit_month_plot[-1][i][-1])  # 输出每个月的实际利润
            profit_total[i].insert(0, profit_total_plot[-1][i][-1])  # 输出截止到当月的总利润
            if Parameter.VirtualShow == 1:
                virtual_profit_month[i].insert(0, virtual_profit_month_plot[-1][i][-1])  # 输出每个月的利润
                virtual_profit_total[i].insert(0, virtual_profit_total_plot[-1][i][-1])  # 输出截止到当月的总利润
                virtual_profit_12_month[i].insert(0, virtual_profit_12_plot[-1][i][-1])  # 输出过去12个月的月均利润
            # 输出截止到当月的总供给
            for crop in range(Parameter.num_crop):
                supply_total[i][crop].insert(0, supply_total_plot[-1][i][crop][-1])# WHY实际作物供给

        profit_12_plot.append(profit_12_month)
        profit_month_plot.append(profit_month)
        profit_total_plot.append(profit_total)
        if Parameter.VirtualShow == 1:
            virtual_profit_month_plot.append(virtual_profit_month)
            virtual_profit_total_plot.append(virtual_profit_total)
            virtual_profit_12_plot.append(virtual_profit_12_month)
        supply_total_plot.append(supply_total)


        x_1 = [i for i in range(x[-1][-1], x[-1][-1] + num_month + 1)]
        x.append(x_1)
        Parameter.Num_Month += num_month
        print("运行至第" + str(Parameter.Num_Month) + "个月")
        #if cooperative_env[0].step_num >= Parameter.Max_month:
        if cooperative_env[0].step_num >= Parameter.Max_month - 1:
            break
    return x, profit_12_plot, virtual_profit_12_plot, profit_month_plot,virtual_profit_month_plot,\
           profit_total_plot,virtual_profit_total_plot,supply_total_plot, label_

#本函数用不能策略的函数，有可能训练不同算法
def cooperative_DynamicCmpare_ForDiffertPolicy(cooperative_env,cooperative_status,Tune_month, pre_month):
    #cooperative_pro 中合作概率为0表示随机，cooperative_pro为1表示完全采用RL
    print("第" + str(Parameter.Num_Month) + "个月开始")
    print('最开始农户随机选择：')
    #Parameter.Num_Month +=  pre_month  #初始的Pre_month月各合作社是一样的策略和动作,Pre_month个月用来预训练。全局变量已经到了第num_month月

    profit_12_month = [[] for i in range(len(cooperative_status))] # 输出过去12个月的月均利润
    profit_month = [[] for i in range(len(cooperative_status))]  # 输出每个月的实际利润
    profit_total =[[] for i in range(len(cooperative_status))]  # 输出截止到当月的总利润
    virtual_profit_total = [[] for i in range(len(cooperative_status))]  # 输出截止到当月的虚拟总利润
    virtual_profit_month = [[] for i in range(len(cooperative_status))]  # 输出每个月的虚拟利润
    virtual_profit_12_month = [[] for i in range(len(cooperative_status))]  # 输出过去12个月的月均虚拟利润
    supply_total = [[[] for i in range(Parameter.num_crop)] for i in range(len(cooperative_status))]  # 输出最近一个月的总供给
    Sum_supply_in_last_12 = [[0.0 for i in range(Parameter.num_crop + 1)] for i in
                             range(len(cooperative_status))]  # 最近12个月的总供给

    #最初一样
    action = []
    #更新价格
    for i in range(len(cooperative_status)):
        if cooperative_status[i].UpdatedPredictPrice == False:  # 不动态更新价格预测算法，根据确定好的算法更新价格列表
            if Parameter.Tune_month_OR_Pre_month == 3:
                cooperative_env[i].Update_PriceList(Parameter.Num_Month + pre_month, 0 , i)  # 价格准备好
                if i == 0:
                    label_.append('price initial ' )
                else:
                    label_.append('price strategy ' + str (i))
            else:
                cooperative_env[i].Update_PriceList(Parameter.Num_Month + pre_month, 0, 0)  # 价格准备好
                label_.append( str(cooperative_status[i].Name()) )

        else:
            if Parameter.Tune_month_OR_Pre_month == 3:
                cooperative_env[i].MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month + pre_month, 0,
                                            i,False)  # 需要从第Parameter.Num_Month个月开始预测
                if i == 0:
                    label_.append('price initial ')
                else:
                    label_.append('price strategy ' + str(i))
            else:
                cooperative_env[i].MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month + pre_month, 0, 0, False)  # 价格准备好
                label_.append(str(cooperative_status[i].Name()) )

        if Parameter.VirtualShow == 2:
            #VirtualMonthPrice[i].append({Parameter.Num_Month: cooperative_env[i].price})
            VirtualMonthPrice[i].update({Parameter.Num_Month: cooperative_env[i].price})


    for j in range(pre_month):
        action_Each_Month = []
        crop_Each_Month = []
        for i in range(len(cooperative_status)):
            if i == 0:
                for f in range(Parameter.num_farmer):
                    # 随机选择动作
                    if cooperative_env[0].is_current_done(cooperative_env[0].farmer_all[f].current_crop, cooperative_env[0].farmer_all[f].plant_month):
                        if Parameter.NormalChoise == False:
                            # action_f = random.randint(1, 13)
                            action_f = Parameter.GlobalRand[Parameter.GlobalRand_index]
                            Parameter.GlobalRand_index += 1
                        else:
                            action_f = Parameter.GlobalRand[Parameter.GlobalRand_index]
                            Parameter.GlobalRand_index += 1

                        cooperative_env[0].farmer_all[f].current_crop = action_f
                        cooperative_env[0].farmer_all[f].plant_month = (j) % 12
                        action.append(action_f)
                        cooperative_env[0].farmer_all[f].FarmerShedule(j, action_f)
                    else:
                        action.append(0)
                        cooperative_env[0].farmer_all[f].FarmerShedule(j, 0)

                    #保存行动数据
                    action_Each_Month.append(cooperative_env[0].farmer_all[f].GetActionShedule(j))
                    crop_Each_Month.append(cooperative_env[0].farmer_all[f].current_crop)

                obs, reward, done, info = cooperative_env[0].step(action)

                # 输出过去12个月的月均利润
                profit_12_month[i].append(round(info['profit'], 2))
                # 输出每个月的实际利润
                profit_month[i].append(round(info['actual_profit'], 2))

                if Parameter.VirtualShow == 1:
                    virtual_profit_12_month[i].append(round(info['profit'], 2))
                    virtual_profit_month[i].append(round(info['actual_profit'], 2))
                # 输出截止到当月的总利润
                if (len(profit_total[i]) == 0) :
                    profit_total[i].append(round(info['actual_profit'], 2))
                    if Parameter.VirtualShow == 1:
                        virtual_profit_total[i].append(round(info['actual_profit'], 2))
                else:
                    profit_total[i].append(profit_total[i][-1] + round(info['actual_profit'], 2))
                    if Parameter.VirtualShow == 1:
                        virtual_profit_total[i].append(profit_total[i][-1] + round(info['actual_profit'], 2))

                for crop in range(Parameter.num_crop):
                    if (len(supply_total[i][crop]) == 0):
                        supply_total[i][crop].append(round(info['supply'][crop + 1], 3))# WHY实际作物供给
                    else:
                        supply_total[i][crop].append(supply_total[i][crop][-1] + round(info['supply'][crop + 1], 3))# WHY实际作物供给
                for crop in range(Parameter.num_crop + 1):
                    # Sum_supply_in_last_12[i][crop] = round(sum(info['supply_in_last_12'][crop]), 3)  # 看最近12个月的作物总供给
                    Sum_supply_in_last_12[i][crop] = round(sum(info['LastYearSupply'][crop]), 3)  # 看最近12个月的作物总供给
            else:
                assert (len(action) != 0)
                cooperative_env[i].step(action[-Parameter.num_farmer:])
                for f in range(Parameter.num_farmer):
                    cooperative_env[i].farmer_all[f].FarmerShedule(j, action[-Parameter.num_farmer:][f])
                    # 保存行动数据
                    # action_Each_Month.append(cooperative_env[0].farmer_all[f].GetActionShedule(j))
                # 输出过去12个月的月均利润
                profit_12_month[i].append(profit_12_month[0][j])
                # 输出每个月的实际利润
                profit_month[i].append(profit_month[0][j])
                # 输出截止到当月的总利润
                profit_total[i].append(profit_total[0][j])

                if Parameter.VirtualShow == 1:
                    virtual_profit_12_month[i].append(profit_12_month[0][j])
                    virtual_profit_month[i].append(profit_month[0][j])
                    virtual_profit_total[i].append(profit_total[0][j])

                # 输出截止到当月的总供给
                for crop in range(Parameter.num_crop):
                    supply_total[i][crop].append(supply_total[0][crop][j])# WHY实际作物供给

                for crop in range(Parameter.num_crop + 1):
                    Sum_supply_in_last_12[i][crop] = (Sum_supply_in_last_12[0][crop])  # 看最近12个月的作物总供给

            # 保存数据
            if Parameter.VirtualShow == 0 or Parameter.VirtualShow == 2 or Parameter.VirtualShow == 3:
                data = [profit_month[i][-1], profit_12_month[i][-1], profit_total[i][-1],
                 round(info['rotation_problem'], 2), round(info['plant_0_yeild'], 2), round(info['action_0'], 2)]
                if Parameter.ConsiderMiniSupply == True:
                    data.append(round(info['MiniSupply_problem'], 2))
                    for crop in range(Parameter.num_crop):
                        data.append(info['MiniSupply_problem_list'][crop + 1])
                SaveProfitFile(data, j, i)
            elif Parameter.VirtualShow == 1:
                data = [profit_month[i][-1],virtual_profit_month[i][-1], profit_12_month[i][-1],virtual_profit_12_month[i][-1], profit_total[i][-1],virtual_profit_total[i][-1],
                            round(info['rotation_problem'], 2), round(info['plant_0_yeild'], 2),round(info['action_0'], 2) ]
                if Parameter.ConsiderMiniSupply == True:
                    data.append(round(info['MiniSupply_problem'], 2))
                    for crop in range(Parameter.num_crop):
                        data.append(info['MiniSupply_problem_list'][crop + 1])
                SaveProfitFile(data, j, i)


            xx = []
            for crop in range(Parameter.num_crop):
                xx.append(supply_total[i][crop][-1])

            if Parameter.ConsiderMiniSupply:  # 看最近12个月的作物总供给
                for crop in range(Parameter.num_crop):
                    xx.append(Sum_supply_in_last_12[i][crop + 1])

            SaveSupplyFile(xx, j, i)
            Crop_total_bar[i].append(action_Each_Month)
            SaveCropRotationFile(action_Each_Month, j, i)
            SaveCurrentCropFile(crop_Each_Month, j, i)


    profit_12_plot.append(profit_12_month)
    profit_month_plot.append(profit_month)
    profit_total_plot.append(profit_total)
    if Parameter.VirtualShow == 1:
        virtual_profit_month_plot.append(virtual_profit_month)
        virtual_profit_total_plot.append(virtual_profit_total)
        virtual_profit_12_plot.append(virtual_profit_12_month)
    supply_total_plot.append(supply_total)


    x_1 = [i for i in range(1, pre_month + 1)]
    x.append(x_1)
    Parameter.Num_Month += pre_month  # 初始的Pre_month月各合作社是一样的策略和动作,Pre_month个月用来预训练。全局变量已经到了第num_month月
    print("运行至第" + str(Parameter.Num_Month) + "个月")
    #开始不同合作社的比较了，或不同参数曲线的比较
    while (True):

        # print("第" + str(Parameter.Num_Month) + "个月开始")
        if cooperative_env[0].step_num + Tune_month >= Parameter.Max_month:
            num_month = Parameter.Max_month - cooperative_env[0].step_num - 1
        else:
            num_month = Tune_month
        # Parameter.Num_Month += num_month


        profit_12_month = [[] for i in range(len(cooperative_env))]  # 输出过去12个月的月均利润
        profit_month = [[] for i in range(len(cooperative_env))]  # 输出每个月的实际利润
        profit_total = [[] for i in range(len(cooperative_env))]  # 输出截止到当月的总利润
        virtual_profit_month = [[] for i in range(len(cooperative_env))]  # 输出截止到当月的虚拟总利润
        virtual_profit_total = [[] for i in range(len(cooperative_env))]  # 输出截止到当月的虚拟总利润
        virtual_profit_12_month = [[] for i in range(len(cooperative_env))]  # 输出过去12个月的月均虚拟利润
        supply_total = [[[] for i in range(Parameter.num_crop)] for i in range(len(cooperative_env))]  # 输出最近一个月的总供给

        print("第" + str(Parameter.Num_Month) + "个月,开始连续" + str(num_month) + "个月运行RL 算法")

        # 连续num_month个月运行RL 算法
        for i in range(len(cooperative_status)):


            cooperative_status[i].ChangeCooperativeStatus()#改变全局变量
            # 配置RL算法
            random_seed = Parameter.Random_Seed
            train_for_Dynamic.setup_seed(random_seed)
            cfg = train_for_Dynamic.DQNConfig()

            if cooperative_status[i].UpdatedRL :#如果比较不同价格就需要不同的算法
                # Parameter.Save_name = cooperative_status[i].Name()
                print("第" + str(i) + "个合作社开始更新RL算法！！！")
                # 更新RL算法
                print("第" + str(Parameter.Num_Month) + "个月开始更新RL算法")

                if Parameter.SelfAdaption_Step_or_data == True:
                    if Parameter.Num_Month >= 70:
                        Parameter.Step_or_data = 1
                        Parameter.Train_Eps = 200  # 训练的回合数
                        Parameter.Max_Step = 120  # 每回合迭代最大步长
                    else:
                        Parameter.Step_or_data = 0
                        Parameter.Train_Eps = 150  # 训练的回合数
                        Parameter.Max_Step = 120  # 每回合迭代最大步长

                writer = SummaryWriter()
                # 训练
                env, agent = train_for_Dynamic.env_agent_config(cfg, seed=Parameter.Random_Seed)
                rewards, ma_rewards = train_for_Dynamic.train_Dynamic(cfg, env, agent,num_month, i)
                make_dir(cfg.result_path, cfg.model_path)
                agent.save(path=cfg.model_path)
                save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
                # plot_rewards(rewards, ma_rewards, tag="train", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
                writer.close()
            else:
                print("第" + str(i) + "个合作社开始未使用DQN")
            supply_lasttotal = []
            for crop in range(Parameter.num_crop):
                supply_lasttotal.append(supply_total_plot[-1][i][crop][-1])
            cooperative_env[i], profit_12_month[i], virtual_profit_12_month[i], profit_month[i], virtual_profit_month[
                i], profit_total[i], virtual_profit_total[i], supply_total[i] = ProbRL_Dynamic_in_next(cooperative_env[i],
                                                                                               cooperative_status[i].cooperative_pro,
                                                                                               num_month,
                                                                                               cooperative_env[
                                                                                                   i]._get_observation(),
                                                                                               cfg.model_path,
                                                                                               profit_total_plot[-1][i][
                                                                                                   -1],
                                                                                               supply_lasttotal, i)


            profit_12_month[i].insert(0, profit_12_plot[-1][i][-1])# 输出过去12个月的月均利润
            profit_month[i].insert(0, profit_month_plot[-1][i][-1])  # 输出每个月的实际利润
            profit_total[i].insert(0, profit_total_plot[-1][i][-1])  # 输出截止到当月的总利润
            if Parameter.VirtualShow == 1:
                virtual_profit_month[i].insert(0, virtual_profit_month_plot[-1][i][-1])  # 输出每个月的利润
                virtual_profit_total[i].insert(0, virtual_profit_total_plot[-1][i][-1])  # 输出截止到当月的总利润
                virtual_profit_12_month[i].insert(0, virtual_profit_12_plot[-1][i][-1])  # 输出过去12个月的月均利润
            # 输出截止到当月的总供给
            for crop in range(Parameter.num_crop):
                supply_total[i][crop].insert(0, supply_total_plot[-1][i][crop][-1])# WHY实际作物供给

        profit_12_plot.append(profit_12_month)
        profit_month_plot.append(profit_month)
        profit_total_plot.append(profit_total)
        if Parameter.VirtualShow == 1:
            virtual_profit_month_plot.append(virtual_profit_month)
            virtual_profit_total_plot.append(virtual_profit_total)
            virtual_profit_12_plot.append(virtual_profit_12_month)
        supply_total_plot.append(supply_total)


        x_1 = [i for i in range(x[-1][-1], x[-1][-1] + num_month + 1)]
        x.append(x_1)

        Parameter.Num_Month += num_month
        print("运行至第" + str(Parameter.Num_Month) + "个月")

        if cooperative_env[0].step_num >= Parameter.Max_month - 1:
            break
    return x, profit_12_plot, virtual_profit_12_plot, profit_month_plot,virtual_profit_month_plot,\
           profit_total_plot,virtual_profit_total_plot,supply_total_plot, label_


def Draw_Profit_Plot(profit_12_plot_, virtual_profit_12_plot_,profit_month_plot_, virtual_profit_month_plot_,profit_total_plot_,virtual_profit_total_plot_):


    if Parameter.VirtualShow == 1:
        l = len(label_)
        for i in range (l):
            label_.append(str(label_[0]) + ' virtuai_profit')
        title_ = ["virtual and actual profit for latest 12 month", "virtual and actual profit for every month", "virtual and actual total profit"]
        simulation_plot.simulation_plot_4_virtual(x, [profit_12_plot_], [virtual_profit_12_plot_], label_,
                                                      'virtual and actual average profit for 12 month')
        # 保存图片名
        decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'virtual_and_actual_average_profit_for 12 month')


        simulation_plot.simulation_plot_4_virtual(x, [profit_month_plot_], [virtual_profit_month_plot_], label_,
                                                      'virtual and actual month profit')
        # 保存图片名
        decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'virtual_and_actual_month_profit')


        simulation_plot.simulation_plot_4_virtual(x, [profit_total_plot_], [virtual_profit_total_plot_], label_,
                                                      'virtual and actual total profit')
        # 保存图片名
        decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'virtual_and_actual_total profit')



    elif Parameter.VirtualShow == 0:
        plt_ = [profit_12_plot_, profit_month_plot_, profit_total_plot_]
        title_ = ["profit for latest 12 month", "profit for every month", "total profit"]
        simulation_plot.simulation_plot_4(x, plt_, label_, title_)

def Draw_Supply_Plot(supply_total_plot_):# 显示各月总供给

    plt.figure()
    for i in range(Parameter.num_cooperative):
        with sns.axes_style('ticks'):  # 使用ticks主题
            plt.subplot(Parameter.num_cooperative, 1, i + 1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.3, hspace=1.4)
            # plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)
            simulation_plot.simulation_plot_5(x, supply_total_plot_, Parameter.vegetable, "supply_total_plot")
            plt.title(label_[i], fontsize=Parameter.MyTitleSize)
            plt.tick_params(labelsize=Parameter.MyFontSize)  # 调整坐标轴数字大小
            if i == Parameter.num_cooperative - 1:
                plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
                plt.ylabel("Supply(10^3 Kg)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
                plt.legend(bbox_to_anchor=(1.01, 0,0,1), loc=3, borderaxespad=0, prop={'size': Parameter.MyLegendSize},ncol=1)
                # plt.legend( prop = {'size':17})   图例字体大小
    plt.tight_layout()
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'Supply')


def Draw_Profit_Histogram(profit_month_plot, label_):# 显示各个合作社每年的总收益
    with sns.axes_style('ticks'):  # 使用ticks主题
        plt.figure()
        profit_month_cooperative = [[] for i in range(Parameter.num_cooperative)]
        profit_year_cooperative = [[] for i in range(Parameter.num_cooperative)]
        for i in range(Parameter.num_cooperative):
            for m in range(len(profit_month_plot)):
                if m == 0:
                    for n in range(len(profit_month_plot[m][i])):
                        profit_month_cooperative[i].append(profit_month_plot[m][i][n])
                else:
                    for n in range(len(profit_month_plot[m][i]) - 1):
                        profit_month_cooperative[i].append(profit_month_plot[m][i][n+1])
        if len(profit_month_cooperative[0]) % 12 == 0:
            num = len(profit_month_cooperative[0]) // 12
        else:
            num = len(profit_month_cooperative[0]) // 12 + 1
        for i in range(Parameter.num_cooperative):
            for j in range( num ):
                if j!= num - 1:
                    profit_year = sum(profit_month_cooperative[i][j * 12: j * 12 + 11])
                else:
                    profit_year = sum(profit_month_cooperative[i][j * 12: ])
                profit_year_cooperative[i].append(profit_year)

            print("num_cooperative is ", i)


        y = [[] for i in range(len(profit_year_cooperative[0]))]
        # width_ = 0.15  # the width of the bars
        width_ = 1 / (Parameter.num_cooperative+1)  # the width of the bars

        for h in range(len(profit_year_cooperative[0])):
            for m in range(len(profit_year_cooperative)):
                y[h].append(profit_year_cooperative[m][h])
        data = np.array(y)
        x_label = []
        x = np.arange(len(profit_year_cooperative[0]))
        for h in range(len(profit_year_cooperative[0])):
            x_label.append(str(h+1) + ' Year')

        for m in range(len(profit_year_cooperative)):
            plt.bar(x + m * width_, height=data[:, m], width=width_, label=str(label_[m]))
        plt.tight_layout()

        plt.title('Profit Histogram of Each Year', fontsize=Parameter.MyTitleSize)
        plt.tick_params(labelsize=Parameter.MyFontSize)  # 调整坐标轴数字大小
        plt.xticks(x, x_label)
        plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': Parameter.MyLegendSize})
        plt.xlabel("Year",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
    plt.tight_layout()
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'Profit_Histogram')


def Draw_Frequency_of_Planting(crop_total_bar_, pre_month,label_):# supply_total_plot 直方图，统计各个合作社的各作物的频次

    plt.figure()
    crop_f_total = [[0 for i in range(Parameter.num_crop)] for i in range(Parameter.num_cooperative)]
    crop_f_part = [[0 for i in range(Parameter.num_crop)] for i in range(Parameter.num_cooperative)]
    x_label = ['potato', 'tomato', 'cucumber', 'pakchoi', 'broccoli', 'cabbage', 'turnip', 'lettuce',
               'wax_gourd', 'bean', 'pepper', 'eggplant', 'celery']
    x = np.arange(len(x_label))  # the label locations
    for i in range(Parameter.num_cooperative):
        with sns.axes_style('ticks'):  # 使用ticks主题
            plt.subplot(Parameter.num_cooperative, 1, i + 1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.3, hspace=1.4)
            sns.despine()  # 在ticks基础上去掉上面的和右边的端线
            # plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)
            for h in range(len(crop_total_bar_[i])):
                for f in range(Parameter.num_farmer):
                    # for crop in range(Parameter.num_crop):
                    # crop_f_total[crop] = crop_total_bar[i].count(crop + 1)
                    # crop_f_part[crop] = crop_total_bar[i][Parameter.Pre_month:].count(crop + 1)
                    if crop_total_bar_[i][h][f] != 0:
                        crop_f_total[i][crop_total_bar_[i][h][f] - 1] += 1
                        if h >= pre_month:
                            crop_f_part[i][crop_total_bar_[i][h][f] - 1] += 1

            print("num_cooperative is ", i)
            print(Parameter.vegetable)
            print("Total Number of Planting for ", crop_f_total[i])
            print("Part Number of Planting for ", crop_f_part[i])
            for a, b in zip(Parameter.vegetable, crop_f_part[i]):
                plt.text(a, b + 1, b, ha='center', va='bottom', fontsize='14')

            simulation_plot.simulation_plot_6(Parameter.vegetable, crop_f_part[i])
            plt.title(label_[i], loc='left',fontsize=Parameter.MyTitleSize)
            if (Parameter.num_cooperative >= 3):
                plt.tick_params(labelsize=12)  # 调整坐标轴数字大小
            else:
                plt.tick_params(labelsize=16)  # 调整坐标轴数字大小
            plt.xticks(x, x_label, rotation=20)

            if i == Parameter.num_cooperative - 2:
                plt.ylabel("Number of Proposed Planting",{'fontname':'Times New Roman','fontsize':20})
            if i == Parameter.num_cooperative - 1:
                plt.xlabel("Crop",{'fontname':'Times New Roman','fontsize':20})

        plt.tight_layout()
        # 保存图片名
        decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'NumberOfProposedPlanting')


    plt.figure()
    for i in range(Parameter.num_cooperative):
        with sns.axes_style('ticks'):  # 使用ticks主题
            plt.subplot(Parameter.num_cooperative, 1, i + 1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.3, hspace=1.4)
            sns.despine()  # 在ticks基础上去掉上面的和右边的端线
            simulation_plot.simulation_plot_6(Parameter.vegetable, crop_f_total[i])
            plt.title(label_[i], loc='left', fontsize=16)
            if (Parameter.num_cooperative >= 3):
                plt.tick_params(labelsize=12)  # 调整坐标轴数字大小
            else:
                plt.tick_params(labelsize=16)  # 调整坐标轴数字大小
            plt.xticks(x, x_label, rotation=20)
            if i == Parameter.num_cooperative - 2:
                plt.ylabel("Number of Total Planting",{'fontname':'Times New Roman','fontsize':20})
            if i == Parameter.num_cooperative - 1:
                plt.xlabel("Crop",{'fontname':'Times New Roman','fontsize':20})

            for a, b in zip(Parameter.vegetable, crop_f_total[i]):#显示坐标轴数字
                plt.text(a, b + 1, b, ha='center', va='bottom', fontsize='14')
        # 保存图片名
        decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'NumberOfTotalPlanting')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def Draw_Frequency_of_Planting_PerMonth(crop_total_bar_, pre_month,label_):# supply_total_plot 直方图，统计各个合作社的各作物的频次
    with sns.axes_style('ticks'):  # 使用ticks主题
        plt.figure()
        crop_f_total = [[[0 for i in range(Parameter.num_crop)] for i in range(12)] for i in range(Parameter.num_cooperative)]
        crop_f_part = [[[0 for i in range(Parameter.num_crop)] for i in range(12)] for i in range(Parameter.num_cooperative)]
        x_label = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        x = np.arange(len(x_label))  # the label locations

        for i in range(Parameter.num_cooperative):
            plt.subplot(Parameter.num_cooperative, 1, i + 1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.3, hspace=1.4)
            # plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)
            for h in range(len(crop_total_bar_[i])):
                for f in range(Parameter.num_farmer):
                    if crop_total_bar_[i][h][f] != 0:
                        crop_f_total[i][h%12][crop_total_bar_[i][h][f] - 1] += 1
                        if h >= pre_month:
                            crop_f_part[i][h%12][crop_total_bar_[i][h][f] - 1] += 1

            print("num_cooperative is ", i)
            #print("label is ", label_[0])
            print(Parameter.vegetable)
            for h in range(12):
                b_array = np.array(crop_f_part[i][h])
                b_index = b_array.argsort()[-3:][::-1]  # 输出前3个最大值的索引
                print("%d month, top 3 vegetables are %s , %s , %s" %(h+1,Parameter.vegetable[b_index[0]], Parameter.vegetable[b_index[1]],Parameter.vegetable[b_index[2]]))
                print("Part Number of Planting for ", crop_f_part[i][h])
            y = [[] for i in range(12)]

            width_ = 0.07  # the width of the bars

            for h in range(12):
                for m in range(len(crop_f_part[i][h])):
                    y[h].append(crop_f_part[i][h][m])
            data = np.array(y)
            for m in range(Parameter.num_crop):
                plt.bar(x + m * width_, height= data[:, m], width = width_, label=str(Parameter.vegetable[m]))
            plt.tight_layout()


            plt.title(label_[i],  loc='left', fontsize=Parameter.MyTitleSize)
            if Parameter.num_cooperative == 1:
                plt.tick_params(labelsize=22)  # 调整坐标轴数字大小
            else:
                plt.tick_params(labelsize=16)  # 调整坐标轴数字大小
            plt.xticks(x, x_label, rotation=20)
            plt.grid(visible=True, axis='x')  # 只显示x轴网格线
            if i == Parameter.num_cooperative - 2:
                plt.ylabel("Number of Proposed Planting per Month",{'fontname':'Times New Roman','fontsize':20})
            if i == Parameter.num_cooperative - 1:
                if Parameter.num_cooperative == 1:
                    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': (Parameter.MyLegendSize + 8)} ,ncol=1)
                else:
                    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': (Parameter.MyLegendSize + 2)}, ncol=1)

        plt.tight_layout()
        # 保存图片名
        decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'NumberOfProposedPlantingPerMonth')

def Draw_Frequency_of_Planting_PerCrop(crop_total_bar_, pre_month,label_):# supply_total_plot 直方图，统计各个合作社的各作物的频次,每种作物
    with sns.axes_style('ticks'):  # 使用ticks主题
        plt.figure()

        crop_f_total = [[[0 for i  in range(12)] for i in range(Parameter.num_crop)] for i in
                        range(Parameter.num_cooperative)]
        crop_f_part = [[[0 for i  in range(12)] for i in range(Parameter.num_crop)] for i in
                        range(Parameter.num_cooperative)]
        x_label = ['potato', 'tomato', 'cucumber', 'pakchoi', 'broccoli', 'cabbage', 'turnip', 'lettuce',
                   'wax_gourd', 'bean', 'pepper', 'eggplant', 'celery']
        month_ = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']
        x = np.arange(len(x_label))  # the label locations
        for i in range(Parameter.num_cooperative):
            plt.subplot(Parameter.num_cooperative, 1, i + 1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.3, hspace=1.4)
            # plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)
            for h in range(len(crop_total_bar_[i])):
                for f in range(Parameter.num_farmer):
                    if crop_total_bar_[i][h][f] != 0:
                        crop_f_total[i][crop_total_bar_[i][h][f] - 1][h%12] += 1
                        if h >= pre_month:
                            crop_f_part[i][crop_total_bar_[i][h][f] - 1][h%12] += 1


            print("num_cooperative is ", i)
            print(Parameter.vegetable)
            for c in range(Parameter.num_crop):
                b_array = np.array(crop_f_part[i][c])
                b_index = b_array.argsort()[-3:][::-1]  # 输出前3个最大值的索引
                print("vegetable is %s , top 3 months are %s , %s , %s" %(Parameter.vegetable[c],month_[b_index[0]], month_[b_index[1]],month_[b_index[2]]))
                print("Part Number of Planting for ", crop_f_part[i][c])
            y = [[] for i in range(Parameter.num_crop)]

            width_ = 0.07  # the width of the bars
            for c in range(Parameter.num_crop):
                for m in range(len(crop_f_part[i][c])):
                    y[c].append(crop_f_part[i][c][m])
            data = np.array(y)
            for m in range(12):
                plt.bar(x + m * width_, height= data[:, m], width = width_, label=str(month_[m]))
            plt.tight_layout()


            plt.title(label_[i], loc='left', fontsize=16)
            if Parameter.num_cooperative == 1:
                plt.tick_params(labelsize=22)  # 调整坐标轴数字大小
            else:
                plt.tick_params(labelsize=16)  # 调整坐标轴数字大小
            plt.grid(visible=True, axis='x')  # 只显示x轴网格线
            ##x倾斜角度
            plt.xticks(x, x_label, rotation=20)
            if i == Parameter.num_cooperative - 2:
                plt.ylabel("Number of Proposed Planting per Crop")
            if i == Parameter.num_cooperative - 1:
                if Parameter.num_cooperative == 1:
                    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': (Parameter.MyLegendSize + 8)}, ncol=1)
                else:
                    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': (Parameter.MyLegendSize + 2)}, ncol=1)
    plt.tight_layout()
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'NumberOfProposedPlantingPerCrop')





def Draw_Diff_Plot_TuneMonth(x_DiffTuneMonth_,profit_12_plot_DiffTuneMonth_, profit_month_plot_DiffTuneMonth_,profit_total_plot_DiffTuneMonth_,
                   supply_total_plot_DiffTuneMonth_,Crop_total_bar_DiffTuneMonth_):
    #p0 = p1= p2= p3 = p4 =[]
    p0 = []


    title_ = ["profit_12_plot", "profit_month_plot", "profit_total_plot"]
    #plt.figure()

    simulation_plot.simulation_plot_7(x_DiffTuneMonth_, profit_12_plot_DiffTuneMonth_,label_1,title_[0])
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, title_[0])
    simulation_plot.simulation_plot_7(x_DiffTuneMonth_, profit_month_plot_DiffTuneMonth_, label_1,title_[1])
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, title_[1])
    simulation_plot.simulation_plot_7(x_DiffTuneMonth_, profit_total_plot_DiffTuneMonth_, label_1,title_[2])
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, title_[2])

def Draw_Diff_Plot_Pre_month(x_DiffPre_month_,profit_12_plot_DiffPre_month_, profit_month_plot_DiffPre_month_,profit_total_plot_DiffPre_month_,
                   supply_total_plot_DiffPre_month_,Crop_total_bar_DiffPre_month_):
    #p0 = p1= p2= p3 = p4 =[]
    p0 = []


    title_ = ["profit_12_plot", "profit_month_plot", "profit_total_plot"]
    #plt.figure()

    simulation_plot.simulation_plot_8(x_DiffPre_month_, profit_12_plot_DiffPre_month_,label_1,title_[0])
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, title_[0])
    simulation_plot.simulation_plot_8(x_DiffPre_month_, profit_month_plot_DiffPre_month_, label_1,title_[1])
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, title_[1])
    simulation_plot.simulation_plot_8(x_DiffPre_month_, profit_total_plot_DiffPre_month_, label_1,title_[2])
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, title_[2])

def Draw_Price_Plot():

    price = [[] for i in range(Parameter.num_crop)]
    intial_price =[]
    res = []
    for i in range(Parameter.num_crop):
        filename = dirname(abspath(__file__)) + '/price_data/宁波数据/月均价格-' + \
                   Parameter.action_name[i + 1] + '.csv'
        # filename = r'../price_data/月均价格-'+self.action_name[i+1]+'.csv'
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            #res.append()
            price[i].append([float(row[2]) for row in reader])

        intial_price.append([v for v in price[i][0]])

    # 价格调整
    for i in range(Parameter.num_crop):
        count = 0
        for p in range (len(Parameter.Price_Change)):
            if Parameter.vegetable[i] == Parameter.Price_Change[p][0]:
                if count >= len(price[i]):
                    price[i].append([v for v in intial_price[i]])
                count += 1
                for m in range(len(Parameter.Price_Change[p][1])):
                    start = Parameter.Price_Change[p][1][m][0]
                    end = Parameter.Price_Change[p][1][m][1] + 1
                    l = len(price[i][-1])
                    if l < start:
                        break
                    elif l >= start and l < end:
                        end = l + 1
                    part = price[i][-1][start:end]
                    price[i][-1][start: end] = [v * (Parameter.Price_Change[p][2][m] + 1) for v in part]
    label_.clear()
    for p in range(len(Parameter.Price_Change)):
        if p == 0:
            label_.append('price initial ' )
        else:
            label_.append('price strategy ' + str (p))

    x = [v for v in range(len(price[0][0]))]
    for i in range(Parameter.num_crop):
        # plt.subplot(3, 5, i + 1)
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
        # plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)
        with sns.axes_style('ticks'):  # 使用ticks主题
            plt.figure()
            #plt.title('Month Price', fontsize=16)
            plt.tight_layout()

            simulation_plot.simulation_plot_9(x, price[i], label_, Parameter.vegetable[i])
            plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.ylabel("Price(Yuan)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            count = 0
            for p in range(len(Parameter.Price_Change)):
                if Parameter.vegetable[i] == Parameter.Price_Change[p][0]:
                    count += 1
            if count > 0:
                plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': 20})
            else:
                plt.legend()
                # 保存图片名
            decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, str(Parameter.vegetable[i]) + '_Price')

def Read_Supply_Data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)#跳过表头
        for row in reader:
            supply = []
            for i in range(Parameter.num_crop):
                supply.append(float(row[1+i]))
            #data.update({int(row[0]):supply})
            data.append(supply)
    return data
def Read_Profit_Data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)#跳过表头
        for row in reader:
            supply = []
            if Parameter.ConsiderMiniSupply == False:
                for i in range(6):
                    supply.append(float(row[1+i]))

            else:
                for i in range(Parameter.num_crop + 7):
                    supply.append(float(row[1+i]))
            data.append(supply)
    return data
def Read_CurrentCrop_Data(filename, SequenceOrGantt):#SequenceOrGatte如果为True，则计算频次，否则计算甘特图
    data = [[] for i in range(Parameter.num_farmer)]
    CurrentCrop_EveryFarmer = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)#跳过表头
        for row in reader:
            CurrentCrop_EveryMonth = []
            for i in range(Parameter.num_farmer):
                CurrentCrop_EveryMonth.append(int(row[1+i]))
            CurrentCrop_EveryFarmer.append(CurrentCrop_EveryMonth)
    for i in range(Parameter.num_farmer):
        data[i].append(CurrentCrop_EveryFarmer[0][i])
        for j in range(len(CurrentCrop_EveryFarmer) ):
            if SequenceOrGantt == True:#计算频次
                if j > 0 and CurrentCrop_EveryFarmer[j][i] != CurrentCrop_EveryFarmer[j-1][i]:
                    data[i].append(CurrentCrop_EveryFarmer[j][i])
            else:#计算甘特图
                if j > 0:
                    data[i].append(CurrentCrop_EveryFarmer[j][i])
    return data


def Draw_Virtual_Profit_Plot(profit_total_plot, VirtualMonthPrice):
    VirtualMonthProfit = [{} for i in range(Parameter.num_cooperative)]
    VirtualTotalProfit = [{} for i in range(Parameter.num_cooperative)]
    Total_Supply = []
    Profit_Data = []
    Rotation_problem = []
    MiniSupply_problem = []
    dir_name = '.\outputs_dynamic\outputs'
    file_names = os.listdir(str(dir_name))#返回该文件夹下所有文件名

    #读取supply数据
    for i in range(Parameter.num_cooperative):
        filename = 'Multi_cooperative' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) + '_Pre_month' + str(Parameter.Pre_month[0]) + '_cooperate' + str(i) + '_Supply_data_' + str(
            Parameter.Random_Seed) + '.csv'
        if filename in file_names:
            Total_Supply.append(Read_Supply_Data(str(dir_name) +'/'+ str(filename)))
        else:
            print('未找到文件！')
    # 读取profit数据
    for i in range(Parameter.num_cooperative):
        filename = 'Multi_cooperative' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) + '_Pre_month' + str(Parameter.Pre_month[0]) + '_cooperate' + str(
            i) + '_Profit_data_' + str(
            Parameter.Random_Seed) + '.csv'
        if filename in file_names:
            Profit_Data.append(Read_Profit_Data(str(dir_name) + '/' + str(filename)))
        else:
            print('未找到文件！')

    for i in range(Parameter.num_cooperative):
        rotation_problem = []
        miniSupply_problem = []
        for j in range(len(Profit_Data[i])):
            rotation_problem.append(Profit_Data[i][j][3])
            if Parameter.ConsiderMiniSupply == True:
                miniSupply_problem.append(Profit_Data[i][j][6])
        Rotation_problem.append(rotation_problem)
        MiniSupply_problem.append(miniSupply_problem)

    #计算收益
    #如果ENV程序中的每个农户也要计算轮作影响f.profit_in_last_12[-1] = f.profit_in_last_12[-1] * (0.8 ** f_rotation_problem_list[i])，那么这里总收益没法加上这个部分计算
    for i in range(Parameter.num_cooperative):
        number = -1#曲线数量
        for k, v in VirtualMonthPrice[i].items():
            number += 1
            total_profit = 0
            total_profit_list = []
            month_profit_list = []
            for j in range(len(Total_Supply[i]) - k):
                month_profit = 0
                if (j == 0):
                    if(number == 0):
                        total_profit = profit_total_plot[number][i][j]
                    else:
                        total_profit = profit_total_plot[number][i][j + 1]
                    if (k == 0):
                        month_profit = 0
                    else:
                        for n in range(Parameter.num_crop):
                            month_profit = (Total_Supply[i][j + k][n] - Total_Supply[i][j + k - 1][n]) * ( v[n][j + k]- Parameter.Plant_Cost[n])
                else:
                    for n in range(Parameter.num_crop):
                        # total_profit += (Total_Supply[i][j+k][n] - Total_Supply[i][j+k-1][n]) * (v[n][j + k] - Parameter.Plant_Cost[n])
                        month_profit += (Total_Supply[i][j+k][n] - Total_Supply[i][j+k-1][n]) * (v[n][j + k] - Parameter.Plant_Cost[n])
                    res = month_profit * (0.8 ** Rotation_problem[i][j+k])
                    if Parameter.ConsiderMiniSupply == True:
                        res = res * (0.8 ** MiniSupply_problem[i][j+k])
                    total_profit += res
                total_profit_list.append(total_profit)
                month_profit_list.append(month_profit)
            VirtualTotalProfit[i].update({k:total_profit_list})
            VirtualMonthProfit[i].update({k:month_profit_list})

    #生成图片
    simulation_plot.simulation_plot_11(profit_total_plot, VirtualTotalProfit)
    # # 保存图片名
    # decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'virtual_and_actual_total_profit')
def CalCulateCurrentCropSequence(CurrentCrop, Order, threshold):#Order表示是连续多少次种植；超过threshold才计算频次
    str_list = []
    substr_list = []
    dic_list = {}
    for i in range (len(CurrentCrop)):
        strr = ','.join(str(e) for e in CurrentCrop[i])
        str_list.append(strr)
        #print(str_list[i])
        for j in range(len(CurrentCrop[i]) - Order + 1):
            substr_list.append(','.join(str(e) for e in CurrentCrop[i][j : j+Order]))
    #去掉重复元素
    unique_substr_list = list(set(substr_list))
    for i in range (len(unique_substr_list)):
        count = 0
        for j in range (len(CurrentCrop)):
            count += str_list[j].count(unique_substr_list[i])
        if count > threshold:
            dic_list.update({unique_substr_list[i]:count})
    return dic_list
def Draw_CurrentCropSequence_Histogram(profit_total_plot, Order, threshold):#画作物序列频次的直方图;Order表示是连续多少次种植；超过threshold才计算频次
    Total_CurrentCrop = []
    CurrentCropSequence = []
    CurrentCropSequence_list = []
    dir_name = '.\outputs_dynamic\outputs'
    file_names = os.listdir(str(dir_name))#返回该文件夹下所有文件名
    #读取CurrentCrop数据
    for i in range(Parameter.num_cooperative):
        filename = 'Multi_cooperative' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) + '_Pre_month' + str(Parameter.Pre_month[0]) + '_cooperate' + str(i) + '_CurrentCrop_data_' + str(
            Parameter.Random_Seed) + '.csv'
        if filename in file_names:
            Total_CurrentCrop.append(Read_CurrentCrop_Data(str(dir_name) +'/'+ str(filename), True))
        else:
            print('未找到文件！')
    #计算作物序列的频次
    for i in range(Parameter.num_cooperative):
        CurrentCropSequence.append(CalCulateCurrentCropSequence(Total_CurrentCrop[i], Order, threshold))
        with sns.axes_style('ticks'):  # 使用ticks主题
            plt.figure()

            plt.title('Cooperate' + str(i + 1), fontsize=16)
            # plt.title('Month Price')
            plt.tight_layout()
            count = 0
            # 设置字体样式和大小
            font = {'family': 'Times New Roman', 'size': 28}
            font_value = {'family': 'Times New Roman', 'size': 20}
            if Parameter.num_cooperative == 1:
                plt.xlabel("Crop Sequence",{'fontname':'Times New Roman','fontsize':25})
                plt.ylabel("Frequency",{'fontname':'Times New Roman','fontsize':25})
            else:
                plt.xlabel("Crop Sequence", {'fontname': 'Times New Roman', 'fontsize': 20})
                plt.ylabel("Frequency", {'fontname': 'Times New Roman', 'fontsize': 20})
            if Parameter.num_cooperative == 1:
                plt.tick_params(labelsize=22)  # 调整坐标轴数字大小
            else:
                plt.tick_params(labelsize=16)  # 调整坐标轴数字大小
            # plt.set_xlabel("Crop Sequence",font)
            # plt.set_ylabel("Times",font)
            #key使用lambda匿名函数取value进行排序
            CurrentCropSequence_list.append(sorted(CurrentCropSequence[i].items(),key = lambda x:x[1],reverse = True))

            N = min(len(CurrentCropSequence_list[i]),  10)
            ###设置柱子宽度
            width_ = 0.45
            x = np.arange(N)
            y_list = [[] for m in range(Order)]
            y_tips_list = [[] for m in range(Order)]
            y_list_name = [[] for m in range(Order)]
            print("Order is" + str(Order) + "and number is " + str(N))
            for j in range(N):
                list_of_str = CurrentCropSequence_list[i][j][0].split(',')
                list_of_integers = [int(m) for m in list_of_str]
                for m in range(len(list_of_integers)):
                    y_list_name[m].append(Parameter.vegetable[list_of_integers[m] - 1])
                    y_tips_list[m].append(CurrentCropSequence_list[i][j][1] / Order)
                    y_list[m].append((m + 1) * CurrentCropSequence_list[i][j][1] / Order)

            for m in range(Order):
                if m == 0:
                    plt.bar(x, y_tips_list[m], width = width_, alpha=0.7)
                elif m == Order - 1 :
                    bar1 = plt.bar(x, y_tips_list[m], width=width_, bottom=y_list[m - 1], alpha=0.7)
                else:
                    plt.bar(x, y_tips_list[m], width = width_,  bottom=y_list[m-1], alpha=0.7)
            # 调整字体颜色、柱子宽度等其他参数，显示坐标轴数字
            count = 0
            for rect, y_value in zip(bar1, y_tips_list[-1]):
                height = rect.get_height()
                for mm in range(Order):
                    if Parameter.num_cooperative == 1:
                        plt.text(rect.get_x() + rect.get_width() / 2., (mm + 0.5) * y_value, y_list_name[mm][count],
                                 ha='center', va='bottom', fontsize=22, fontname='Times New Roman')
                    else:
                        plt.text(rect.get_x() + rect.get_width() / 2., (mm+0.5) * y_value , y_list_name[mm][count],
                         ha='center', va='bottom', fontsize=16,  fontname='Times New Roman')
                count += 1
                if Parameter.num_cooperative == 1:
                    plt.text(rect.get_x() + rect.get_width() / 2., y_value * Order, y_value * Order,
                             ha='center', va='bottom', fontsize=22, fontname='Times New Roman')
                else:
                    plt.text(rect.get_x() + rect.get_width() / 2., y_value * Order, y_value * Order,
                             ha='center', va='bottom', fontsize=16, fontname='Times New Roman')


            # 保存图片名
            decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, str(i) + '_CurrentCropSequence_Order' + str(Order))




def Cal_CurrentCrop_GanttChart(CurrentCrop):#计算作物甘特图
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, Parameter.num_crop + 1))
    AllFarmer_barh = []
    AllColors = []
    for i in range(Parameter.num_farmer):
        farmer_barh = []
        facecolors = []
        cycle = 1
        for j in range(len(CurrentCrop[i])):
            if j == 0:
                crop = CurrentCrop[i][j]
                start_month = j
            else:
                if (CurrentCrop[i][j] == CurrentCrop[i][j - 1]):
                    cycle += 1
                else:
                    farmer_barh.append((start_month,cycle))
                    facecolors.append(category_colors[crop])
                    cycle = 1
                    crop = CurrentCrop[i][j]
                    start_month = j
                if j == len(CurrentCrop[i]) - 1:
                    farmer_barh.append((start_month, cycle))
                    facecolors.append(category_colors[crop])
        AllFarmer_barh.append(farmer_barh)
        AllColors.append(facecolors)
    return AllFarmer_barh, AllColors,category_colors

def Cal_CurrentCrop_GanttChart_2(CurrentCrop):#计算作物甘特图
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, Parameter.num_crop + 1))
    AllFarmer_barh = []
    AllColors = []
    flag = 0 #如果有action=0的情况，那么flag = 1
    for i in range(Parameter.num_farmer):
        farmer_barh = []
        facecolors = []
        cycle = 1
        for j in range(len(CurrentCrop[i])):
            if j == 0:
                crop = CurrentCrop[i][j]
                start_month = j
            else:
                if (CurrentCrop[i][j] == CurrentCrop[i][j - 1]):
                    cycle += 1
                else:
                    farmer_barh.append((start_month,cycle))
                    if crop == 0:
                        facecolors.append([0.498, 0.498, 0.498, 1.0])  # 如果action为0，颜色为灰色
                        flag = 1
                    else:
                        facecolors.append(category_colors[crop])
                    # facecolors.append(category_colors[crop])
                    cycle = 1
                    crop = CurrentCrop[i][j]
                    start_month = j
                if j == len(CurrentCrop[i]) - 1:
                    farmer_barh.append((start_month, cycle))
                    if crop == 0:
                        flag = 1
                    facecolors.append(category_colors[crop])
                    # facecolors.append(category_colors[crop])
        AllFarmer_barh.append(farmer_barh)
        AllColors.append(facecolors)
    return AllFarmer_barh, AllColors,category_colors, flag
def Draw_CurrentCrop_GanttChart():#画作物甘特图
    Total_CurrentCrop = []
    CurrentCropSequence = []
    CurrentCropSequence_list = []
    dir_name = '.\outputs_dynamic\outputs'
    file_names = os.listdir(str(dir_name))  # 返回该文件夹下所有文件名
    # 读取CurrentCrop数据
    for i in range(Parameter.num_cooperative):
        filename = 'Multi_cooperative' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) + '_Pre_month' + str(Parameter.Pre_month[0]) + '_cooperate' + str(
            i) + '_CurrentCrop_data_' + str(
            Parameter.Random_Seed) + '.csv'
        if filename in file_names:
            Total_CurrentCrop.append(Read_CurrentCrop_Data(str(dir_name) + '/' + str(filename),False))
        else:
            print('未找到文件！')
    #计算作物甘特图
    for i in range(Parameter.num_cooperative):
        # AllFarmer_barh, AllColors,category_colors = Cal_CurrentCrop_GanttChart(Total_CurrentCrop[i]) # 计算作物甘特图
        AllFarmer_barh, AllColors,category_colors,flag = Cal_CurrentCrop_GanttChart_2(Total_CurrentCrop[i]) # 计算作物甘特图
        # plt.figure()
        with sns.axes_style('ticks'):  # 使用ticks主题
            fig, ax = plt.subplots()
            plt.title('Cooperate' + str(i + 1), fontsize=Parameter.MyTitleSize)
            # plt.title('Month Price')
            plt.tight_layout()
            count = 0
            # 设置字体样式和大小
            font = {'family': 'Times New Roman', 'size': 28}
            font_value = {'family': 'Times New Roman', 'size': 20}
            plt.tick_params(labelsize=Parameter.MyFontSize)  # 调整坐标轴数字大小
            plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.ylabel("Farmers",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})


            ax.invert_yaxis()
            # 设置xy轴的范围
            ax.set_ylim(0, 265)
            ax.set_xlim(0, Parameter.Max_month - 1)

            # 更改y轴记号标签
            labels = []
            for f in range(Parameter.num_farmer):
                if (f + 1) % 10 == 0:
                    if f < 9:
                        labels.append('F0' + str(f + 1))
                    else:
                        labels.append('F' +str(f+1))
                else:
                    labels.append('')
            ax.set_yticks(np.linspace(10, 255, (Parameter.num_farmer)), labels=labels)

            # 设置条形数据
            for f in range(Parameter.num_farmer):
                ax.broken_barh(AllFarmer_barh[f], (5*f + 8, 4), facecolors=AllColors[f])
                # plt.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), facecolors=('tab:orange', 'tab:green', 'tab:red'))
            # plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': 7})
        #     ax.legend(ncol=len(category_colors), bbox_to_anchor=(1, 1),  loc='lower left', fontsize='small')

            # 添加图例
            category_colors[0] = [0.498,0.498,0.498,1.0]#如果action为0，颜色为灰色
            patches = [mpatches.Patch(color=category_colors[m], label="{:s}".format(Parameter.action_name[m])) for m in range(len(Parameter.action_name))]


            plt.legend(handles=patches, bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': (Parameter.MyLegendSize+2)})
            # plt.legend(handles=patches,loc='upper center', borderaxespad=0, prop={'size': 16}, ncol = 7)
            # 保存图片名
            # decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'CurrentCrop_GanttChart_' + str(i) )
        plt.tight_layout()
def Draw_CurrentCrop_GanttChart_2():#画作物甘特图
    Total_CurrentCrop = []
    CurrentCropSequence = []
    CurrentCropSequence_list = []
    dir_name = '.\outputs_dynamic\outputs'
    file_names = os.listdir(str(dir_name))  # 返回该文件夹下所有文件名
    # 读取CurrentCrop数据
    for i in range(Parameter.num_cooperative):
        filename = 'Multi_cooperative' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) + '_Pre_month' + str(Parameter.Pre_month[0]) + '_cooperate' + str(
            i) + '_CurrentCrop_data_' + str(
            Parameter.Random_Seed) + '.csv'
        if filename in file_names:
            Total_CurrentCrop.append(Read_CurrentCrop_Data(str(dir_name) + '/' + str(filename),False))
        else:
            print('未找到文件！')
    #计算作物甘特图
    with sns.axes_style('ticks'):  # 使用ticks主题
        Fig = plt.figure()
        for i in range(Parameter.num_cooperative):
            # AllFarmer_barh, AllColors,category_colors = Cal_CurrentCrop_GanttChart(Total_CurrentCrop[i]) # 计算作物甘特图
            AllFarmer_barh, AllColors,category_colors,flag = Cal_CurrentCrop_GanttChart_2(Total_CurrentCrop[i]) # 计算作物甘特图
            ax = Fig.add_subplot(2, 2, i+1)

            plt.tight_layout()
            count = 0
            # 设置字体样式和大小
            font = {'family': 'Times New Roman', 'size': 28}
            font_value = {'family': 'Times New Roman', 'size': 20}
            plt.tick_params(labelsize=Parameter.MyFontSize)  # 调整坐标轴数字大小
            plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.ylabel("Farmers",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})


            ax.invert_yaxis()
            # 设置xy轴的范围
            ax.set_ylim(0, 265)
            ax.set_xlim(0, Parameter.Max_month - 1)

            # 更改y轴记号标签
            labels = []
            for f in range(Parameter.num_farmer):
                if (f + 1) % 10 == 0:
                    if f < 9:
                        labels.append('F0' + str(f + 1))
                    else:
                        labels.append('F' + str(f + 1))
                else:
                    labels.append('')
            ax.set_yticks(np.linspace(10, 255, (Parameter.num_farmer)), labels=labels)

            # 设置条形数据
            for f in range(Parameter.num_farmer):
                ax.broken_barh(AllFarmer_barh[f], (5*f + 8, 4), facecolors=AllColors[f])


            # 添加图例
            if i == Parameter.num_cooperative - 1 :
                category_colors[0] = [0.498, 0.498, 0.498, 1.0]  # 如果action为0，颜色为灰色
                patches = [mpatches.Patch(color=category_colors[m], label="{:s}".format(Parameter.action_name[m])) for m in range(len(Parameter.action_name))]
                Fig.legend(handles=patches, loc='upper center', borderaxespad=0, prop={'size': (Parameter.MyLegendSize+1.5)}, ncol=7)
    # 保存图片名
    decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'CurrentCrop_GanttChart_Total' )
    # plt.savefig(str(Parameter.IMAGES_PATH) + 'CurrentCrop_GanttChart_' + str(i), dpi=250, bbox_inches='tight')

def Draw_MiniSupply_Plot():
    Total_Supply = []
    Supply_EveryFarmer = [[ [] for j in range(Parameter.num_crop)]for i in range(Parameter.num_cooperative)]
    Supply_res = [[] for j in range(Parameter.num_crop)]
    Profit_Data = []
    Rotation_problem = []
    Plant_0_yeild_problem = []
    Action_0_problem = []
    MiniSupply_problem = []
    dir_name = '.\outputs_dynamic\outputs'
    file_names = os.listdir(str(dir_name))  # 返回该文件夹下所有文件名
    # 读取supply数据
    for i in range(Parameter.num_cooperative):
        filename = 'Multi_cooperative' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) + '_Pre_month' + str(Parameter.Pre_month[0]) + '_cooperate' + str(
            i) + '_Supply_data_' + str(
            Parameter.Random_Seed) + '.csv'
        if filename in file_names:
            Total_Supply.append(Read_Supply_Data(str(dir_name) + '/' + str(filename)))
        else:
            print('未找到文件！')
    for i in range(Parameter.num_cooperative):
            for j in range(len(Total_Supply[i]) ):#j个月
                for m in range(len(Total_Supply[i][j])):#m种作物
                    if j % 12 == 0:
                        Supply_res[m] = Total_Supply[i][j][m]
                    Supply_EveryFarmer[i][m].append(Total_Supply[i][j][m] - Supply_res[m])



    # 读取profit数据
    for i in range(Parameter.num_cooperative):
        filename = 'Multi_cooperative' + str(Parameter.Save_name) + '_Tune_month' + str(
            Parameter.Tune_month[0]) + '_Pre_month' + str(Parameter.Pre_month[0]) + '_cooperate' + str(
            i) + '_Profit_data_' + str(
            Parameter.Random_Seed) + '.csv'
        if filename in file_names:
            Profit_Data.append(Read_Profit_Data(str(dir_name) + '/' + str(filename)))
        else:
            print('未找到文件！')

    for i in range(Parameter.num_cooperative):
        rotation_problem = []
        miniSupply_problem = []
        plant_0_yeild_problem =[]
        action_0_problem = []
        for j in range(len(Profit_Data[i])):
            rotation_problem.append(Profit_Data[i][j][3])
            plant_0_yeild_problem.append(Profit_Data[i][j][4])
            action_0_problem.append(Profit_Data[i][j][5])
            if Parameter.ConsiderMiniSupply == True:
                miniSupply_problem.append(Profit_Data[i][j][6])
        Rotation_problem.append(rotation_problem)
        MiniSupply_problem.append(miniSupply_problem)
        Plant_0_yeild_problem.append(plant_0_yeild_problem)
        Action_0_problem.append(action_0_problem)
    #画最小供给的变化曲线
    num = sum(i>0 for i in Parameter.Min_Demand)
    index = [i for i, e in enumerate(Parameter.Min_Demand) if e != 0]
    x = [i for i in range(Parameter.Max_month - 1)]
    for i in range(Parameter.num_cooperative):
        # plt.GridSpec(4, 2, wspace=0.2, hspace=0.5)
        if Parameter.ConsiderMiniSupply == True:
            with sns.axes_style('ticks'):  # 使用ticks主题
                plt.figure()
                for j in range(num):
                    plt.subplot(num, 1, j + 1)
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.5)
                    plt.plot(x, Supply_EveryFarmer[i][index[j] - 1], color='green', label=str(Parameter.action_name[index[j]]), linestyle='-', linewidth=2)
                    plt.hlines(Parameter.Min_Demand[index[j]], 0, Parameter.Max_month - 1, color="red",label= 'MiniDemand for ' + str(Parameter.action_name[index[j]]),linestyle='--', linewidth=1)
                    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': 20})
                    if j == 0:
                        plt.title('Cooperate' + str(i + 1), fontsize=16)
                    if j == num - 1:
                        plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
                        plt.ylabel("Supply(10^3 Kg)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
                # 保存图片名
                decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'MiniSupply_' + str(i + 1))
                # plt.savefig(str(Parameter.IMAGES_PATH) + 'MiniSupply_' + str(i + 1), dpi=400, bbox_inches='tight')

    # 画问题的变化曲线


    if Parameter.ConsiderMiniSupply == True:
        N = 4
    else:
        N = 3

    label =[]
    for i in range(Parameter.num_cooperative):
        label.append('Cooperate_' + str(i + 1))
    # label = ['Nomal_Cooperation','Nomal_Cooperation']
    plt.figure()
    with sns.axes_style('ticks'):  # 使用ticks主题
        plt.subplot(N, 1, 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.5)
        simulation_plot.simulation_plot_9(x, Rotation_problem, label, 'rotation_problem')
        plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': Parameter.MyLegendSize})

    with sns.axes_style('ticks'):  # 使用ticks主题
        plt.subplot(N, 1, 2)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.5)
        simulation_plot.simulation_plot_9(x, Plant_0_yeild_problem, label, 'plant_0_yeild_problem')
        plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': Parameter.MyLegendSize})
    with sns.axes_style('ticks'):  # 使用ticks主题
        plt.subplot(N, 1, 3)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.5)
        simulation_plot.simulation_plot_9(x, Action_0_problem, label, 'action_0_problem')
        plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': Parameter.MyLegendSize})

        if Parameter.ConsiderMiniSupply == True:
            with sns.axes_style('ticks'):  # 使用ticks主题
                plt.subplot(N, 1, 4)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.5)
                simulation_plot.simulation_plot_9(x, MiniSupply_problem, label, 'miniSupply_problem')
                plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': Parameter.MyLegendSize})

if __name__ == "__main__":

    random.seed(Parameter.Random_Seed)
    for i in range(1000000):
        Parameter.GlobalRand.append(random.randint(1, 13))
    profit_12_plot_DiffTunePara = []  # 用来放不同的profit_12_plot的结果，比如在不同的Tune_month、Pre_month参数下
    profit_month_plot_DiffTunePara = []  # 用来放不同的profit_month_plot的结果，比如在不同的Tune_month、Pre_month参数下
    profit_total_plot_DiffTunePara = []  # 用来放不同的profit_total_plot的结果，比如在不同的Tune_month、Pre_month参数下
    supply_total_plot_DiffTunePara = []  # 用来放不同的supply_total_plot的结果，比如在不同的Tune_month、Pre_month参数下
    Crop_total_bar_DiffTunePara = []  # 用来放不同的Crop_total_bar的结果，比如在不同的Tune_month、Pre_month参数下
    x_DiffTunePara = []# 用来放不同的x的结果，比如在不同的Tune_month、Pre_month参数下
    label_1 = []
    cooperative_status = []  # 合作社不同策略
    # Where to save the figures
    os.makedirs(Parameter.IMAGES_PATH, exist_ok=True)
    #是调整Tune_month或Pre_month参数
    if Parameter.Tune_month_OR_Pre_month == 1 and len(Parameter.Tune_month) > 1 and len(Parameter.Pre_month) == 1:  # 调整超参数Tune_month
        para_num = len(Parameter.Tune_month)
        for i in range(len(Parameter.Tune_month)):
            if i!= 0:
                Parameter.Pre_month.append(Parameter.Pre_month[0])
            for j in range(Parameter.num_cooperative):
                if Parameter.num_cooperative == 1:
                    str_ = 'Tune_month is ' + str(Parameter.Tune_month[i])
                else:
                    str_ = 'cooperative_pro is ' + str(Parameter.cooperative_pro[j]) + '; Tune_month is ' + str(
                    Parameter.Tune_month[i])
                label_1.append(str_)
    elif Parameter.Tune_month_OR_Pre_month == 2 and len(Parameter.Tune_month) == 1 and len(Parameter.Pre_month) > 1: #调整超参数Pre_month
        para_num = len(Parameter.Pre_month)
        for i in range(len(Parameter.Pre_month)):
            if i!= 0:
                Parameter.Tune_month.append(Parameter.Tune_month[0])
            for j in range(Parameter.num_cooperative):
                if Parameter.num_cooperative == 1:
                    str_ = 'Pre_month is ' + str(Parameter.Pre_month[i])
                else:
                    str_ = 'cooperative_pro is ' + str(Parameter.cooperative_pro[j]) + '; Pre_month is ' + str(Parameter.Pre_month[i])
                label_1.append(str_)
    elif Parameter.Tune_month_OR_Pre_month == 0 and len(Parameter.Tune_month) == 1 and len(Parameter.Pre_month) == 1:
        para_num = 1
        print("Multi cooperative compare!")
        assert (len(Parameter.cooperative_pro) >= Parameter.num_cooperative)
        if Parameter.SamePolicyDifferetPro == False:
            # id, UpdatedRL, UpdatedPredictPrice, cooperative_pro,NormalChoise,CosiderPrice,ConsiderMiniSupply,n_state, n_actions
            if Parameter.ConsiderMiniSupply == True:
                cooperative_status.append(CooperativePolicy(0, False,False, 0.0, False, False, True,0,0))  # 随机策略
                cooperative_status.append(CooperativePolicy(1, False, False,0.0, True, False, True,0,0))  # 正常考虑轮作策略，考虑最低供给需求
                cooperative_status.append(CooperativePolicy(2, False,False, 0.0, True, True, True,0,0))  # 正常考虑轮作和市场价格策略，考虑最低供给需求
                cooperative_status.append(CooperativePolicy(3, True, True,1.0, True, True, True,(Parameter.num_crop * 4 + 1 + 4),Parameter.num_crop + 1) )# DQN策略，考虑最低供给需求
            else:
                cooperative_status.append(CooperativePolicy(0, False,False, 0.0, False, False, False,0,0))  # 随机策略
                cooperative_status.append(CooperativePolicy(1, False, False,0.0, True, False, False,0,0))  # 正常考虑轮作策略
                cooperative_status.append(CooperativePolicy(2, False,False, 0.0, True, True, False,0,0))  # 正常考虑轮作和市场价格策略
                cooperative_status.append(CooperativePolicy(3, True, True,1.0, True, True, False,(Parameter.num_crop * 3 + 1 + 3), Parameter.num_crop + 1) ) # DQN策略，不考虑最低供给需求
            Parameter.num_cooperative = len(cooperative_status)  # 合作社数量
    elif Parameter.Tune_month_OR_Pre_month == 3:
        print("Multi price compare!")
        #para_num = len(Parameter.Price_Change)
        para_num = 1
        assert (len(Parameter.Price_Change) == Parameter.num_cooperative)
        assert (len(Parameter.cooperative_pro) == Parameter.num_cooperative)
    else:
        print("Tune_month or Pre_month config wrong!")
        assert (0)
    if Parameter.VirtualShow == 2:
        print('VirtualShow = 2')
        #VirtualMonthPrice = [[] for i in range(Parameter.num_cooperative)]
        VirtualMonthPrice = [{} for i in range(Parameter.num_cooperative)]
    elif Parameter.VirtualShow == 3:
        print('VirtualShow = 3')
        VirtualTotalProfit = [{} for i in range(Parameter.num_cooperative)]
    for i in range(para_num):
        # 价格预测模型训练
        # TrainPricePredict_GBDT(Parameter.History_Month)
        Parameter.Num_Month = 0
        if Parameter.Tune_month_OR_Pre_month == 1:
            Parameter.Save_name =  Parameter.Tune_month[i]
        elif Parameter.Tune_month_OR_Pre_month == 2:
            Parameter.Save_name = Parameter.Pre_month[i]
        elif Parameter.Tune_month_OR_Pre_month == 3:
            Parameter.Save_name =''#后面程序改了，这不能改
        else:
            Parameter.Save_name =''
        cooperative = [[] for i in range(Parameter.num_cooperative)]#合作社，注意合作社规模相当，都是num_farmer个社员
        cooperative_state = [[] for i in range(Parameter.num_cooperative)]#合作社状态

        cooperative_env = [[] for i in range(Parameter.num_cooperative)]  # 合作社环境
        cooperative_obs = [[] for i in range(Parameter.num_cooperative)]  # 合作社观察
        actionPick = ActionPick()

        profit_test = []
        rotation_problem_test = []
        plant_0_yeild_test = []
        action_0 = []
        supply = []
        supply_12 = []

        x = []

        profit_12_plot = []  # 输出过去12个月的月均利润
        profit_month_plot = []  # 输出每个月的实际利润
        profit_total_plot = []  # 输出每个月的总实际利润
        virtual_profit_month_plot = []  # 输出每个月的虚拟利润
        virtual_profit_total_plot = []  # 输出每个月的总虚拟利润
        virtual_profit_12_plot = []  # 输出过去12个月的月均虚拟利润
        supply_total_plot = []  # 输出截止到当月的总供给
        Crop_total_bar = [[] for i in range(Parameter.num_cooperative)]  # 各合作社种的作物频次的直方图



        label_ = []

        for f in range(Parameter.num_farmer):#初始化这几个合作社都有初始一样的农户和种植
            res_1 = round(random.random() + 1, 2)
            res_2 = random.randint(1, 13)
            for j in range(Parameter.num_cooperative):
                cooperative[j].append(farmer(f, res_1,res_2))
                cooperative_state[j].append(res_2)

        for j in range(Parameter.num_cooperative):
            cooperative_env[j] = MyEnv(cooperative[j], cooperative_state[j])
            cooperative_obs[j] = cooperative_env[j].reset()
        if Parameter.ConsiderMiniSupply ==True:
            #n_states = cooperative_env[0].num_crop * 4 + 1 + 3  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
            Parameter.n_states = cooperative_env[0].num_crop * 4 + 1 + 4  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
        else:
            Parameter.n_states = cooperative_env[0].num_crop * 3 + 1 + 3  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
            #n_states = cooperative_env[0].num_crop * 3 + 1 + 6  # WHY 各种作物在当前月份下种植时的预期销售价格、预期产量、预期生育期长度；
        # WHY 当前月份；农户的种植面积、当前种植作物、当前作物的种植时间
        Parameter.n_actions = cooperative_env[0].num_crop + 1

        #每Tune_month个月调整一次
        # 合作社比较或某一个合作社运行，重新跑一遍程序RL模型并保存
        if Parameter.SamePolicyDifferetPro == True:
            x, profit_12_plot, virtual_profit_12_plot, profit_month_plot,virtual_profit_month_plot,\
            profit_total_plot,virtual_profit_total_plot, supply_total_plot, label_ = \
                cooperative_DynamicCmpare(cooperative_env, Parameter.cooperative_pro, cooperative_obs,
                                          Parameter.Tune_month[i],Parameter.Pre_month[i])
        else:


            x, profit_12_plot, virtual_profit_12_plot, profit_month_plot, virtual_profit_month_plot, \
            profit_total_plot, virtual_profit_total_plot, supply_total_plot, label_ = \
            cooperative_DynamicCmpare_ForDiffertPolicy(cooperative_env, cooperative_status,
                                      Parameter.Tune_month[i], Parameter.Pre_month[i])
        profit_12_plot_DiffTunePara.append(profit_12_plot)
        profit_month_plot_DiffTunePara.append(profit_month_plot)
        profit_total_plot_DiffTunePara.append(profit_total_plot)
        supply_total_plot_DiffTunePara.append(supply_total_plot)
        Crop_total_bar_DiffTunePara.append(Crop_total_bar)
        x_DiffTunePara.append(x)

    if Parameter.LargeFontSize == True:
        Parameter.MyFontSize = 20#设置输出的图的字体较大
        Parameter.MyLegendSize = 14  # 设置输出的图的图例字体大小
        Parameter.MyTitleSize = 16  # 设置输出的图的图例题目大小
    else:
        Parameter.MyFontSize = 9
        Parameter.MyLegendSize = 7  # 设置输出的图的图例字体大小
        Parameter.MyTitleSize = 14  # 设置输出的图的图例题目大小
    if Parameter.Tune_month_OR_Pre_month == 0 or Parameter.Tune_month_OR_Pre_month == 3:
        # label_ =['a','b','c','d']
        Draw_Profit_Plot(profit_12_plot, virtual_profit_12_plot, profit_month_plot, virtual_profit_month_plot, profit_total_plot,virtual_profit_total_plot)
        # Draw_Supply_Plot(supply_total_plot)
        # Draw_Profit_Histogram(profit_month_plot, label_)
        # Draw_Frequency_of_Planting(Crop_total_bar,Parameter.Pre_month[i],label_)
        Draw_Frequency_of_Planting_PerMonth(Crop_total_bar,Parameter.Pre_month[i],label_)
        Draw_Frequency_of_Planting_PerCrop(Crop_total_bar,Parameter.Pre_month[i],label_)
        Draw_CurrentCropSequence_Histogram(profit_total_plot, 2, 1)
        # Draw_CurrentCropSequence_Histogram(profit_total_plot, 3, 1)
        # Draw_CurrentCropSequence_Histogram(profit_total_plot, 4, 1)
        # Draw_CurrentCropSequence_Histogram(profit_total_plot, 5, 1)
        # Draw_CurrentCropSequence_Histogram(profit_total_plot, 6, 1)
        if Parameter.Tune_month_OR_Pre_month == 3:
            Draw_Price_Plot()
    elif Parameter.Tune_month_OR_Pre_month == 1:
        Draw_Diff_Plot_TuneMonth(x_DiffTunePara,profit_12_plot_DiffTunePara,profit_month_plot_DiffTunePara,
                                 profit_total_plot_DiffTunePara,supply_total_plot_DiffTunePara,
                                 Crop_total_bar_DiffTunePara)

        for i in range(len(Crop_total_bar_DiffTunePara)):
            label_.clear()
            label_.append('Tune_month is '+ str(Parameter.Tune_month[i]))
            Draw_Frequency_of_Planting(Crop_total_bar_DiffTunePara[i], Parameter.Pre_month[i],label_)
            Draw_Frequency_of_Planting_PerMonth(Crop_total_bar_DiffTunePara[i], Parameter.Pre_month[i],label_)
            Draw_Frequency_of_Planting_PerCrop(Crop_total_bar_DiffTunePara[i], Parameter.Pre_month[i],label_)
    elif Parameter.Tune_month_OR_Pre_month == 2:
        Draw_Diff_Plot_Pre_month(x_DiffTunePara, profit_12_plot_DiffTunePara, profit_month_plot_DiffTunePara,
                                 profit_total_plot_DiffTunePara, supply_total_plot_DiffTunePara,
                                 Crop_total_bar_DiffTunePara)

        for i in range(len(Crop_total_bar_DiffTunePara)):
            label_.clear()
            label_.append('Pre_month is ' + str(Parameter.Pre_month[i]))
            Draw_Frequency_of_Planting(Crop_total_bar_DiffTunePara[i], Parameter.Pre_month[i],label_)
            Draw_Frequency_of_Planting_PerMonth(Crop_total_bar_DiffTunePara[i], Parameter.Pre_month[i],label_)
            Draw_Frequency_of_Planting_PerCrop(Crop_total_bar_DiffTunePara[i], Parameter.Pre_month[i],label_)
    else:
        print("Tune_month or Pre_month config wrong!")


    if Parameter.VirtualShow == 2:
        Draw_Virtual_Profit_Plot(profit_total_plot, VirtualMonthPrice)

    if Parameter.VirtualShow == 3:
        simulation_plot.simulation_plot_11(profit_total_plot, VirtualTotalProfit)
    # plt.get_current_fig_manager().full_screen_toggle()
    # plt.get_current_fig_manager().window.showMaximized()

    # Draw_CurrentCrop_GanttChart()
    Draw_CurrentCrop_GanttChart_2()
    # Draw_MiniSupply_Plot()
    plt.show()





