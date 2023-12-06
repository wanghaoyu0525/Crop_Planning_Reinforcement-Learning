# core.Env是gym的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
# 必须要重写的方法有:
# __init__()：构造函数
# reset()：初始化环境
# step()：环境动作,即环境对agent的反馈
# render()：如果要进行可视化则实现

import random
import csv

import Parameter
from price_data import decompose_GBDT_monthPredict

from price_data.decompose_GBDT_monthPredict import PricePredict_GBDT

import time
from os.path import dirname, abspath
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

# random.seed(Parameter.Random_Seed)

# from environment.pricePredict import pricePredict


class MyEnv:
    '''
    描述：
        农户在满足轮作约束条件下进行作物选择，保证最低供给的情况下实现利润最大化

    observation:
        当前的价格
        每种作物采收当月的预测价格和产量
        当前所有农户的种植情况——最低供给约束

    actions:
        每个农户所选择的作物
        0表示保持不动

    reward：
        本月采收作物所带来的收益——具有延时性
        违反约束所带来的惩罚

    起始状态：
        农户随机选择作物

    终止条件：
        **顺序种植来自同一植物family的作物；
        回合中步长超过200；

    问题解决的要求：
        **在100个连续回合中return超过？？

    '''

    # TruePrice_month表示需要多少个月的真实数据；pre_month表示需要预测几个月的数据，注意和Parameter.Future_Month二者取最大,num_cooperate第几个价格有所调整
    def Update_PriceList(self, TruePrice_month, pre_month,num_cooperate):
        time_start = time.perf_counter()  # 记录开始时间
        print('Update_PriceList start!')
        self.price.clear()
        for i in range(self.num_crop):
            filename = dirname(abspath(__file__)).replace("environment", "") + 'price_data/宁波数据/月均价格-' + \
                       self.action_name[i + 1] + '.csv'
            # filename = r'../price_data/月均价格-'+self.action_name[i+1]+'.csv'
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                self.price.append([float(row[2]) for row in reader])

            if Parameter.Tune_month_OR_Pre_month == 3:  # 价格变化
                # 价格调整
                if Parameter.vegetable[i] == Parameter.Price_Change[num_cooperate][0]:
                    for m in range(len(Parameter.Price_Change[num_cooperate][1])):
                        start = Parameter.Price_Change[num_cooperate][1][m][0]
                        end = Parameter.Price_Change[num_cooperate][1][m][1] + 1
                        l = len(self.price[i])
                        if l < start:
                            break
                        elif l >= start and l < end:
                            end = l + 1
                        part = self.price[i][start:end]
                        self.price[i][start: end] = [v * (Parameter.Price_Change[num_cooperate][2][m] + 1) for v in part]

            if Parameter.UpdatedPredictPrice == True:
                # 只读当前月份Num_Month之前的数据进行训练
                l = len(self.price[i])
                del self.price[i][TruePrice_month:]
                print("使用了 %d 个月的真实数据" % TruePrice_month)
                if TruePrice_month < Parameter.History_Month:
                    print("价格数据量不足，不能预测未来数据, 需要 %d 个月的数据" % Parameter.History_Month)
                    for j in range(Parameter.Future_Month + 1 - TruePrice_month):  # 根据预测算法代替真实价格数据,需要未来Future_Month个数据
                        self.price[i].append(0.0)
                else:
                    max_ = max(Parameter.Future_Month, pre_month)  # 用2者最大预测
                    print("预测并使用了 %d 个月的数据" % (max_))
                    model_address = '宁波数据/'
                    for j in range(max_):  # 根据预测算法代替真实价格数据
                        self.price[i].append(
                            (PricePredict_GBDT(Parameter.vegetable[i], self.price[i][-Parameter.History_Month:],
                                               model_address)))

        time_end = time.perf_counter()  # 记录开始时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        # for i in range(self.num_crop):
        #     print("price[" + str(i) + "] length is " +str(len(self.price[i])))
        print('Update_PriceList end!')
        print('Update_PriceList运行时间为', time_sum)

    # TruePrice_month表示需要多少个月的真实数据；pre_month表示需要预测几个月的数据，注意和Parameter.Future_Month二者取最大
    # 同decompose_GBDT_monthPredict.py的Update_PriceList_ForTest
    def Update_PriceList_MixedData(self, TruePrice_month, pre_month,num_cooperate, data_address, model_address):
        time_start = time.perf_counter()  # 记录开始时间
        print('Update_PriceList_MixedData start!')
        self.price.clear()
        from os.path import dirname, abspath
        for i in range(self.num_crop):
            filename = dirname(abspath(__file__)).replace("environment", "") + '/' + str(data_address) + '月均价格-' + \
                       Parameter.action_name[i + 1] + '.csv'
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                self.price.append([float(row[2]) for row in reader])

            if Parameter.Tune_month_OR_Pre_month == 3:  # 价格变化
                # 价格调整
                if Parameter.vegetable[i] == Parameter.Price_Change[num_cooperate][0]:
                    for m in range(len(Parameter.Price_Change[num_cooperate][1])):
                        start = Parameter.Price_Change[num_cooperate][1][m][0]
                        end = Parameter.Price_Change[num_cooperate][1][m][1] + 1
                        l = len(self.price[i])
                        if l < start:
                            break
                        elif l >= start and l < end:
                            end = l + 1
                        part = self.price[i][start:end]
                        self.price[i][start: end] = [v * (Parameter.Price_Change[num_cooperate][2][m] + 1) for v in part]

            # 只读当前月份Num_Month之前的数据进行训练
            #l = len(self.price[i])
            if Parameter.UpdatedPredictPrice  == False:
                del self.price[i][TruePrice_month:]
                print("使用了 %d 个月的真实数据" % TruePrice_month)
                if TruePrice_month < Parameter.History_Month:
                    print("价格数据量不足，不能预测未来数据, 需要 %d 个月的数据" % Parameter.History_Month)
                    for j in range(Parameter.Future_Month + 1 - TruePrice_month):  # 根据预测算法代替真实价格数据,需要未来Future_Month个数据
                        self.price[i].append(0.0)
                else:
                    max_ = max(Parameter.Future_Month, pre_month)  # 用2者最大预测
                    print("预测并使用了 %d 个月的数据" % (max_))
                    for j in range(max_):  # 根据预测算法代替真实价格数据
                        self.price[i].append(
                            (PricePredict_GBDT(Parameter.vegetable[i], self.price[i][-Parameter.History_Month:],
                                               model_address)))
            else:
                #使用了其它地方数据
                if TruePrice_month < 10:
                    del self.price[i][10:]
                    print("使用了%s 的 %d 个月的真实数据" % (data_address, 10))
                else:
                    del self.price[i][TruePrice_month:]
                    print("使用了%s 的 %d 个月的真实数据" %(data_address, TruePrice_month))
                print("使用了%s 的 模型" % (model_address))
                max_ = max(Parameter.Future_Month, pre_month)  # 用2者最大预测
                print("预测并使用了 %d 个月的数据" % (max_))
                for j in range(max_):  # 根据预测算法代替真实价格数据
                    self.price[i].append(
                        (PricePredict_GBDT(Parameter.vegetable[i], self.price[i][-Parameter.History_Month:],
                                           model_address)))
        time_end = time.perf_counter()  # 记录开始时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        # for i in range(self.num_crop):
        #     print("price[" + str(i) + "] length is " +str(len(self.price[i])))
        print('Update_PriceList_MixedData end!')
        print('Update_PriceList运行时间为', time_sum)




    #同decompose_GBDT_monthPredict.py的MixedRegionalDataPrediction
    def MixedRegionalDataPrediction(self, history_month, start_month_predict,num_month,num_cooperate,Is_Train):  # 混合数据预测
        # history_month为训练用参数，根据history_month个月数据预测下个月数据
        # start_month_predict为根据TruePrice_month月的数据训练好模型后，从start_month_predict个月开始预测模型
        # TruePrice_month为用多少个月的数据开始训练
        # UseData_address为使用哪里的数据开始预测
        # 北京和宁波的模型训练，对同一组宁波数据进行预测；Traindata_address为训练数据
        # model_address为采用哪里的模型
        # num_month个月更新算法
        # Is_Train如果为True,表示从train_Dynamic过来的，要考虑Step_or_data参数的影响
        time_start = time.perf_counter()  # 记录开始时间
        print('MixedRegionalDataPrediction start!')
        if start_month_predict < 10:#宁波数据量不够，使用北京数据
            UseData_address = 'price_data/新发地数据/'
            TrainData_address = [['price_data/新发地数据/']]
            model_address = '新发地数据/'
            TruePrice_month = [120]
        elif start_month_predict < 20:#宁波数据量不够，使用混合数据
            UseData_address = 'price_data/宁波数据/'
            TrainData_address = [['price_data/新发地数据/', 'price_data/宁波数据/']]
            model_address = '混合数据模型/'
            TruePrice_month = [120, start_month_predict]
        else:#宁波数据量足够，使用宁波数据
            UseData_address = 'price_data/宁波数据/'
            TrainData_address = [['price_data/宁波数据/']]
            model_address = '宁波数据/'
            TruePrice_month = [start_month_predict]

        # 训练算法,如果Parameter.Tune_month_OR_Pre_month == 3，那么价格发生变化，训练数据相应调整
        decompose_GBDT_monthPredict.TrainPricePredict_GBDT_AccodingMixedData(history_month, TruePrice_month, num_cooperate, TrainData_address)



        if Is_Train == False: #不是训练时候的价格预测和更新
            if Parameter.VirtualShow == 0:
                self.Update_PriceList_MixedData(start_month_predict, 0, num_cooperate, UseData_address,
                                         model_address)#预测num_month个月价格准备好
            elif Parameter.VirtualShow == 1 or Parameter.VirtualShow == 2 or Parameter.VirtualShow == 3:
                self.Update_PriceList_MixedData(start_month_predict, num_month + 8, num_cooperate, UseData_address,
                                                model_address)# 预测num_month或8个月数据
        else:
            if Parameter.Step_or_data == 1:
                self.Update_PriceList_MixedData(start_month_predict, 0, num_cooperate, UseData_address,
                                                model_address)  # 预测num_month或8个月数据
            elif Parameter.Step_or_data == 0:
                self.Update_PriceList_MixedData(start_month_predict, Parameter.Max_Step-Parameter.Num_Month+Parameter.Future_Month, num_cooperate, UseData_address,
                                                model_address)  # 预测num_month或8个月数据


        time_end = time.perf_counter()  # 记录开始时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

        print('MixedRegionalDataPrediction end!')
        print('MixedRegionalDataPrediction运行时间为', time_sum)



    def __init__(self, farmer_all, init_state):
        self.farmer_all = farmer_all

        self.current_month = 0
        #self.total_month = 0
        self.step_num = 0

        #self.step_length = 200
        self.step_length = Parameter.Num_Month + Parameter.Future_Month

        self.action_name = Parameter.action_name

        self.num_crop = len(self.action_name) - 1
        self.num_farmer = len(farmer_all)

        self.supply = [0 for i in range(self.num_crop+1)]

        self.planting_length = [0, 3, 5, 4, 2, 4, 4, 3, 3, 3, 2, 4, 5, 3]

        # 每个作物的在过去12个月的最低供给，单位是吨，就是10^3kg
        #self.min_demand = [0, 1.75, 3.5, 2.5, 0.8, 0.8, 2, 2.75, 2, 4.5, 1, 2.75, 2.5, 2.75]
        # 最近12个月整个合作社的西红柿最低供给量为3.5吨/亩*10亩，黄瓜最低供给量为2吨/亩*10亩，生菜最低供给量为2吨/亩*20亩，青椒最低供给量为2吨/亩*10亩
        #[‘不种植’，'土豆', '西红柿', '黄瓜', '菠菜', '菜花', '洋白菜', '白萝卜', '生菜', '冬瓜', '豆角', '青椒', '茄子', '芹菜']
        #self.min_demand = [0, 0, 35, 20, 0, 0, 0, 0, 40, 0, 0, 20, 0, 0]
        #self.min_demand = [0, 0, 20, 10, 0, 0, 0, 0, 20, 0, 0, 10, 0, 0]
        self.min_demand = [ i for i in Parameter.Min_Demand]
        #表示作物还有的剩余的需求
        #self.last_demand = [ i for i in Parameter.Min_Demand]


        # 单位已转化为CNY/kg #WHY?表4-2，用上了，参考文献Designing price-contingent vegetable rotation schedules__using agent-based simulation
        #未来应该改为CNY/亩更合适，这样当产量为0时候，也有投入，收益可能为负
        self.plant_cost = [1, 1.72, 1.63, 1.29, 1.5, 0.67, 0.42, 1.25, 1.07, 2.94, 1.94, 1.71, 0.86]#13位

        # 根据GreenLab模型结果修改
        self.yeild = Parameter.yeild

        self.current_price = []#14位

        self.profit_in_last_12 = []
        self.supply_in_last_12 = [[] for i in range(self.num_crop+1)]
        self.LastYearSupply = [[] for i in range(self.num_crop + 1)]  # 最近的自然年的供给

        # 初始状态
        self.init_state = init_state

        # 从文件中读取所有的历史价格数据
        self.price = []
        #self.MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month,0,0)
        if Parameter.UpdatedPredictPrice == False:
            address = '宁波数据/'
            decompose_GBDT_monthPredict.TrainPricePredict_GBDT(Parameter.History_Month,address)
            self.Update_PriceList(Parameter.Num_Month, 0, 0)
        else:
            self.MixedRegionalDataPrediction(Parameter.History_Month, Parameter.Num_Month, 0, 0,False)
    def reset(self):
        self.profit_in_last_12 = []
        self.supply_in_last_12 = [[] for i in range(self.num_crop + 1)]
        num = 0
        for i in self.farmer_all:
            i.current_crop = self.init_state[num]
            num += 1
            i.plant_month = 0
        # 需要斟酌一下初始的current_month是0还是1
        # WHY 计算下个月用的？
        #self.current_month = 1
        self.current_month = 0
        self.supply = [0 for i in range(self.num_crop + 1)]
        self.step_num = 0
        self._set_current_price(0)
        obs = self._get_observation()
        return obs

    # 让环境进入某一指定状态
    def set_condition(self, c_month, area, c_crop, c_plant_month, profit_in_last_12):
        self.profit_in_last_12 = profit_in_last_12
        self.current_month = c_month
        num = 0
        for i in self.farmer_all:
            i.area = area[num]
            i.current_crop = c_crop[num]
            i.plant_month = c_plant_month[num]
            num += 1
        self._set_current_price(c_month)
        obs = self._get_observation()
        return obs
    def set_current_month(self,c_month):
        self.current_month = c_month
    def step(self, action):
        done = False
        info = {}
        info['rotation_problem'] = 0
        info['plant_0_yeild'] = 0
        info['action_0'] = 0
        info['MiniSupply_problem'] = 0
        Last_supply = 0
        Future_supply = 0
        reward_pre = [0 for i in range(self.num_farmer)]

        rotation_problem = 0 #WHY:轮作问题次数
        plant_0_yeild = 0 #WHY:产量为0的次数
        action_0 = 0 #WHY:不种植次数
        reward_s = 0
        MiniSupply_problem = 0 #不满足最小产量的问题
        MiniSupply_problem_list = [0 for i in range(self.num_crop + 1)]



        self.supply = [0.0 for i in range(self.num_crop + 1)]
        f_rotation_problem_list = [0 for i in range(self.num_farmer)]  # WHY:轮作问题次数
        f_discount_list = [0 for i in range(self.num_farmer)]  # WHY:轮作问题次数

        for i in range(self.num_farmer):#先算所有农户的产量，更新合作社的supply_in_last_12和LastYearSupply
            f = self.farmer_all[i]
            f_yeild = self.yeild[f.current_crop][f.plant_month][(self.current_month) % 12]
            if ( f_yeild == -1):
                f_yeild == 0
            self.supply[f.current_crop] += f.area * f_yeild

            #每个农户最近12个月种的产量
            for crop in range(0, self.num_crop + 1):
                if crop == f.current_crop:
                    if len(f.supply_in_last_12[f.current_crop]) < 12:
                        f.supply_in_last_12[f.current_crop].append(f.area * f_yeild)
                    else:
                        f.supply_in_last_12[f.current_crop].pop(0)
                        f.supply_in_last_12[f.current_crop].append(f.area * f_yeild)
                else:
                    if len(f.supply_in_last_12[crop]) < 12:
                        f.supply_in_last_12[crop].append(0.0)
                    else:
                        f.supply_in_last_12[crop].pop(0)
                        f.supply_in_last_12[crop].append(0.0)

            if Parameter.ConsiderMiniSupply == True:
                if Parameter.YearOrMonthSupply == True:
                    if (self.current_month) % 12 == 0:
                        for crop in range(self.num_crop + 1):
                            f.LastYearSupply[crop].clear()
                    for crop in range(self.num_crop + 1):
                        if crop == f.current_crop:
                            f.LastYearSupply[crop].append(f.area * f_yeild)
                        else:
                            f.LastYearSupply[crop].append(0.0)

            # 计算每个农户的月收益——用于计算产生reward
            f_profit = 0
            for crop in range(1, len(self.supply)):
                # 这里supply为吨，也就是10^3公斤，价格是元/kg，所以收益本身就是10^3元
                f_profit += f.supply_in_last_12[crop][-1] * (self.current_price[crop] - self.plant_cost[
                    crop - 1])  # current_price、supply14位，price、plant_cost13位
            # 将过去12个月的profit存储起来
            if len(f.profit_in_last_12) < 12:
                f.profit_in_last_12.append(f_profit)
            else:
                f.profit_in_last_12.pop(0)
                f.profit_in_last_12.append(f_profit)


        if len(self.supply_in_last_12[0]) < 12:
            for n in range(self.num_crop + 1):
                self.supply_in_last_12[n].append(self.supply[n])
        else:
            for n in range(self.num_crop + 1):
                self.supply_in_last_12[n].pop(0)
                self.supply_in_last_12[n].append(self.supply[n])
        info['supply_in_last_12'] = self.supply_in_last_12
        if Parameter.ConsiderMiniSupply == True:
            if Parameter.YearOrMonthSupply == True:
                # if (self.current_month) % 12 == 0:
                if (self.current_month) % 12 == 11:  # 一般12月份就要备下一年的轮作信息了
                    for crop in range(self.num_crop + 1):
                        self.LastYearSupply[crop].clear()

                for crop in range(self.num_crop + 1):
                    self.LastYearSupply[crop].append(self.supply[crop])

        info['LastYearSupply'] = self.LastYearSupply

        for i in range(self.num_farmer):#再算农户的决策
            f = self.farmer_all[i]
            if Parameter.ConsiderMiniSupply ==True:

                not_satisify = []
                LastSupply_list = []
                not_satisify_count = 0
                demand_crop = []
                for crop in range(1, self.num_crop + 1):
                    if self.min_demand[crop] > 0:
                        demand_crop.append(crop)
                for crop in range(1, self.num_crop + 1):

                    #任务分解到每个农户身上，乘以一个系数，因为不是每个人都正好有机会去种这个，所以要有冗余

                    if self.min_demand[crop] > 0:
                        not_satisify_count += 1
                        if Parameter.YearOrMonthSupply == False:  # 如果是稳定按月供给
                            Last_supply = 0
                            if (len(self.supply_in_last_12[crop]) < Parameter.Order_Month):
                                if (len(self.supply_in_last_12[crop]) !=0):
                                    # Last_supply = sum(self.supply_in_last_12[crop]) / len(self.supply_in_last_12[crop])
                                    Last_supply = sum(self.supply_in_last_12[crop])
                            else:
                                for j in range(1, Parameter.Order_Month + 1):
                                    Last_supply += (self.supply_in_last_12[crop][-j])
                                # Last_supply /= Parameter.Order_Month
                            m_d = 1 * (self.min_demand[crop] - Last_supply)/ self.num_farmer
                            Future_supply = 0
                            if self.is_current_done(f.current_crop, f.plant_month):
                                act = action[i]
                                plant_month = (self.current_month) % 12
                            else:
                                act = f.current_crop
                                plant_month = f.plant_month
                            if crop == act:
                                for m in range(0, Parameter.Order_Month + 1):
                                    supply = f.area * self.yeild[act][plant_month][(self.current_month + m) % 12]
                                    if supply >= 0:#去除等于-1的产量
                                        Future_supply += supply
                                Future_supply /= Parameter.Order_Month


                        elif Parameter.YearOrMonthSupply == True:  # 如果是按照年供给
                            m_d = 1 * (self.min_demand[crop] - sum(self.LastYearSupply[crop])) / self.num_farmer
                            # m_d = 1 * (self.min_demand[crop] - sum(self.LastYearSupply[crop]))
                            Future_supply = 0
                            if self.is_current_done(f.current_crop, f.plant_month):
                                act = action[i]
                                plant_month = (self.current_month) % 12
                            else:
                                act = f.current_crop
                                plant_month = f.plant_month
                            if crop == act:
                                # for m in range(0, 12-(self.current_month) % 12):
                                for m in range(1, 8):#最大生育期不超过8，要不把以前的又重复计算了
                                    supply = f.area * self.yeild[act][plant_month][
                                        (self.current_month + m) % 12]
                                    if supply >= 0:#去除等于-1的产量
                                        Future_supply += supply



                        if Future_supply < m_d and m_d > 0:

                            not_satisify.append((m_d - Future_supply) / m_d)
                            LastSupply_list.append((m_d - Future_supply) / m_d)
                            MiniSupply_problem_list[crop] += 1

                        else:
                            not_satisify.append(0)#满足一个需求是1个需求
                            LastSupply_list.append(0)
                            demand_crop.remove(crop)



                reward_s = 0
                if  len(not_satisify) > 0:
                    if 0 not in not_satisify:#只要有1个需求满足就认为reward_s可以，否则太难满足所有要求了
                        reward_s = -1 * sum(not_satisify)/len(not_satisify)  #转换为[-1,0)
                    else:
                        reward_s = 0
                    f_discount_list[i] = sum(not_satisify) / len(not_satisify)  # 转换为(0,1]

            reward_rotation = 0
            # 前后顺序种植相同植物family作物
            if (action[i] in [1, 2, 11, 12]) and (f.current_crop in [1, 2, 11, 12]):
                rotation_problem += 1
                f_rotation_problem_list[i] += 1
                reward_rotation = -1
                #reward_pre[i] = -1
                #info['rotation_problem'].append('顺序种植solanaceae种族作物')
            elif (action[i] in [4, 5, 6, 7]) and (f.current_crop in [4, 5, 6, 7]):
                rotation_problem += 1
                f_rotation_problem_list[i] += 1
                reward_rotation = -1
                #reward_pre[i] = -1
                #info['rotation_problem'].append('顺序种植brassicaceae种族作物')
            elif (action[i] in [3, 9]) and (f.current_crop in [3, 9]):
                rotation_problem += 1
                f_rotation_problem_list[i] += 1
                reward_rotation = -1
                #reward_pre[i] = -1
                #info['rotation_problem'].append('顺序种植cucurbitaceae种族作物')


            reward_ActionWrong = 0
            reward_ActionNULL = 0
            if self.is_current_done(f.current_crop,f.plant_month):
                f.current_crop = action[i]
                f.plant_month = (self.current_month)%12
                if action[i] == 0:#WHY?:不种植
                    action_0 += 1
                    reward_ActionNULL = -1
                    #reward_pre[i] = -1
                elif sum(self.yeild[f.current_crop][f.plant_month]) == -1:#WHY:或该作物在当前月份种植时不能正常生长（产量 0）时
                    plant_0_yeild += 1
                    reward_ActionWrong = -1
                    #reward_pre[i] = -1
            elif action[i] == 0:#WHY:生育期没结束时候不采取动作
                reward_ActionWrong = 0
                #reward_pre[i] = 0

            if Parameter.ConsiderMiniSupply == True:
                # reward_pre[i] = (0.4*reward_s + 0.2*reward_rotation + 0.4*reward_ActionWrong)#[-1,0]
                # reward_pre[i] = min((0.7*reward_s + 0.3*reward_rotation), reward_ActionWrong)#[-1,0]
                reward_pre[i] = min(reward_s,reward_rotation, reward_ActionWrong,reward_ActionNULL)
                # reward_pre[i] = min(reward_s, reward_ActionWrong)
            else:
                #reward_pre[i] = (reward_rotation + reward_ActionWrong)/2#[-1,0]
                reward_pre[i] = min(reward_rotation, reward_ActionWrong,reward_ActionNULL)

        info['rotation_problem'] = rotation_problem
        info['plant_0_yeild'] = plant_0_yeild
        info['action_0'] = action_0
        #info['MiniSupply_problem'] = MiniSupply_problem
        info['MiniSupply_problem_list'] = MiniSupply_problem_list

        # WHY:获取当前的所有农户的总供给量
        info['supply'] = self.supply

        # 获取当月的市场价格
        self._set_current_price(self.step_num)
        info['price'] = self.current_price

        # 计算合作社的月收益——用于计算产生reward
        profit = 0
        for s in range(1,len(self.supply)):
            #这里supply为吨，也就是10^3公斤，价格是元/kg，所以收益本身就是10^3元
            profit += self.supply[s]*(self.current_price[s]-self.plant_cost[s-1])#current_price、supply14位，price、plant_cost13位
            #profit += self.supply[i] * self.current_price[i]


        discount = 0
        not_satisify = []
        if Parameter.ConsiderMiniSupply == True:
            for crop in range(1, self.num_crop + 1):
                m_d = self.min_demand[crop]
                Last_supply = 0
                if Parameter.YearOrMonthSupply == False:
                    if (len(self.supply_in_last_12[crop]) < Parameter.Order_Month ):
                        #Last_supply = sum(self.supply_in_last_12[crop])

                        # for j in range(1, len(self.supply_in_last_12[crop]) + 1):
                            # Last_supply += (self.supply_in_last_12[crop][-j] - j * m_d / Parameter.Order_Month)
                        if (len(self.supply_in_last_12[crop]) !=0):
                            Last_supply = sum(self.supply_in_last_12[crop]) / len(self.supply_in_last_12[crop])
                            #Last_supply = sum(self.supply_in_last_12[crop])
                    else:
                        # for j in range(1, Parameter.Order_Month + 1):
                            # Last_supply += (self.supply_in_last_12[crop][-j] - j * m_d / Parameter.Order_Month)
                        for j in range(1, Parameter.Order_Month + 1):
                            Last_supply += (self.supply_in_last_12[crop][-j])
                        Last_supply /= Parameter.Order_Month
                elif Parameter.YearOrMonthSupply == True:
                    Last_supply = sum(self.LastYearSupply[crop])

                if Last_supply < 0:
                    Last_supply = 0

                if Last_supply < m_d:

                    not_satisify.append((m_d - Last_supply) / m_d)
                    MiniSupply_problem += 1

            if len(not_satisify) == 0:
                discount = 0
                reward_s = 0
            else:
                discount = sum(not_satisify)/len(not_satisify) #(0,1] ,1为不满足
                reward_s = -sum(not_satisify) / len(not_satisify)
            #profit = profit * (0.8**discount)#指数函数
        info['MiniSupply_problem'] = MiniSupply_problem

        # 农户的轮作、没满足需求问题
        total_profit = 0.0
        for i in range(self.num_farmer):
            f = self.farmer_all[i]
            # 顺序种植相同植物种族的作物会对收益造成影响
            f.profit_in_last_12[-1] = f.profit_in_last_12[-1] * (0.8 ** f_rotation_problem_list[i])

            total_profit += f.profit_in_last_12[-1]
        #如果用所有农户的收益之和计算总合作收益则这个保留（下面那行注释），否则按照合作社的方式计算总合作社的profit则把下面这行注释掉
        profit = total_profit
        # 顺序种植相同植物种族的作物会对收益造成影响
        profit = profit * (0.8 ** rotation_problem)
        if Parameter.ConsiderMiniSupply == True:
            profit = profit * (0.8 ** discount)  # 指数函数整个合作社没完成任务，要利润缩减

        # 将过去12个月的profit存储起来
        if len(self.profit_in_last_12) < 12:
            self.profit_in_last_12.append(profit)
        else:
            self.profit_in_last_12.pop(0)
            self.profit_in_last_12.append(profit)
        info['profit_in_last_12'] = self.profit_in_last_12
        info['profit'] = sum(self.profit_in_last_12)/len(self.profit_in_last_12)
        info['actual_profit'] = profit


        #WHY:计算回报，reward_pre如果有轮作问题、地空着或该作物在当前月份种植时不能正常生长（产量 0）时，取值-1；见Line301-325
        #reward, MiniSupply_problem, MiniSupply_problem_list = self._get_reward(self.profit_in_last_12, self.supply_in_last_12, reward_pre)
        reward = self._get_reward(self.profit_in_last_12, self.supply_in_last_12, self.farmer_all, reward_pre)


        self.current_month = (self.current_month+1)%12
        obs = self._get_observation()
        self.step_num += 1

        return obs, reward, done, info

    def _get_observation(self):
        # 获取所有农户的状态
        obs = []


        # 当前各种作物在当月种植时预期的价格、产量、生育期长度，并根据数据的变化范围将其转化为-1~1
        for i in range(1,self.num_crop+1):
            total_yeild = sum(self.yeild[i][self.current_month %12])+1
            harvest_time = self.yeild[i][self.current_month%12].index(-1) - 1
            #WHY? 这三行公式和毕业论文Page55的预期价格和生育期长度公式反了。产量最大值应该是5，预期产量转换公式不对.
            obs.append(total_yeild * 0.5 - 1.25)

            inteval_month = harvest_time - self.current_month
            if inteval_month < 0:
                inteval_month += 12

            assert (inteval_month <= Parameter.Future_Month)
            if Parameter.UpdatedRL == True:
                if (self.step_num + inteval_month) > len(self.price[i - 1]):
                    print('请查看UpdatedPredictPrice参数是否设置为True或者Pre_month大于10')
                    assert (0)
                obs.append((self.price[i - 1][(self.step_num + inteval_month - 1)] * 10 - 47) / 40)  # 收获月份价格
                obs.append(((harvest_time + 12 - self.current_month)%12-5)/3)
            else:#其实不起作用
                obs.append((self.price[i - 1][(self.step_num - 1)] * 10 - 47) / 40)  # 收获月份价格
                obs.append(((harvest_time + 12 - self.current_month) % 12 - 5) / 3)

        # 当前月份
        #WHY:根据数据的变化范围将其转化为 -1~1
        obs.append(self.current_month*2 / 11 - 1)
        return obs

    def _set_current_price(self, step_num):
        self.current_price = [0]
        for i in range(self.num_crop):
            # self.current_price.append(self.price[i][step_num % 120])
            self.current_price.append(self.price[i][step_num])

    def _get_reward(self, profit_in_last_12, supply_in_last_12, farmer_all, reward_pre):
        MiniSupply_problem = 0
        # 稳定供应的要求,#满足最近的12个月有个最小的供应量需求，否则回报为负的
        not_satisify = []
        # 合作社利润最大化的要求，二次分配到各户
        profit = sum(profit_in_last_12)/len(profit_in_last_12)/self.num_farmer
        if profit > 1.5:
            reward_p = 1
        elif profit < 0.5:
            reward_p = -1
        else:
            reward_p = (profit*2 - 2)/1

        reward_mix = reward_p

        reward = []
        for i in range(len(reward_pre)):
            if reward_pre[i] == 0:
                reward.append(reward_mix)
            else:
                reward.append(reward_pre[i])

        return reward

    # 未完成则返回False，已完成则返回True #WHY?有可能是空着没种东西的？
    '''def is_current_done(self, current_crop, plant_month):
        if self.yeild[current_crop][plant_month][(self.current_month+1)%12] == -1:
            return True
        else:
            return False'''
    def is_current_done(self, current_crop, plant_month):
        if self.yeild[current_crop][plant_month][(self.current_month+1)%12] == -1 or current_crop == 0 or sum(self.yeild[current_crop][plant_month]) == -1:#3
            return True
        else:
            return False

    def get_current_month(self):
        return self.current_month

    def get_current_price(self):
        return self.current_price

    def get_price(self, crop, num):
        if num < len(self.price[crop]):
            return self.price[crop][num]
        else:
            print('获取价格超出范围！！！')
            return 0

    def get_cost(self, crop):
        assert(crop in range(0,13))
        return self.plant_cost[crop]
    def get_yeild(self):
        result = []
        for i in range(1,self.num_crop+1):
            result.append(sum(self.yeild[i][self.current_month])+1)
        return result
