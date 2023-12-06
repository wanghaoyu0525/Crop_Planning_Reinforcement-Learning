import datetime
from dateutil.relativedelta import relativedelta

############################################################################################################
#Config Parameters
SamePolicyDifferetPro = True# 当为True时表示用同一训练好的模型不同概率的合作社策略进行分析比较；为False时表示训练不同模型或者不同策略用于合作社之间比较
ConsiderMiniSupply = True#True 表示_get_reward在计算的时候考虑满足最近的12个月有个最小的供应量需求，同时考虑利润最大化的要求；否则表示只考虑利润最大化的要求
VirtualShow = 0#，如果是0表示不显示虚拟的收益；如果是1表示显示虚拟的收益,但是随着算法更新会把后面的虚拟收益抹掉；如果是2表示显示虚拟的收益，但是不抹掉后面的虚拟收益;
                # 如果是3表示显示虚拟的收益，但是不抹掉后面的虚拟收益，3和2的区别在于，3要每隔Tune_month更新RL时，根据当前更新的RL以及预测的价格预测未来收益。而2不会根据更新的RL，仅仅依靠预测价格来计算虚拟收益
############################################################################################################

UpdatedRL = True #是否更新RL算法，如果是True表示更新算法，如果是false表示不更新算法
NormalChoise = True#如果是True表示农户有点知识来选择下一个动作，如果是false表示农户随机选择动作
CosiderPrice = True#当NormalChoise = True时有效,CosiderPrice是True表示农户考虑当前可种的品种中上个月哪个价格最高来选择下一个动作，如果是false表示农户不考虑价格因素
UpdatedPredictPrice = True#是否更新价格预测算法，如果是True表示更新算法；如果是false表示不更新算法，但是Pre_month必须大于等于10
SelfAdaption_Step_or_data = True#True表示自适应调节Step_or_data参数，即当数据量较小少于70周期时，Step_or_data = 0；当数据量大于70个周期可以训练时候，Step_or_data = 1
YearOrMonthSupply =True#当ConsiderMiniSupply为True时起作用。True表示按照自然年1-12月满足供给；False表示按照连续月的稳定供给
LargeFontSize = True# 当为True时，表示输出的图的字体较大，用于文章使用。正常的选择False即可
############################################################################################################
# Model Parameters
num_farmer = 50 #合作社里农户的数量
GlobalRand = []#全局随机数池子
GlobalRand_index = 0#全局随机数池子的索引
num_crop = 13 #作物的种类
num_cooperative = 1 # 合作社数量
n_states = 0 #算法状态维数，程序中确定
n_actions = 0 #算法动作维数，程序中确定
Max_month = 121 #不超过运行121个月

cooperative_pro = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]  # 合作社好好合作的社员的概率，0表示随机，1表示完全是RL，第一个合作社是参考
MyFontSize = 9 #设置输出的图的字体较大
MyLegendSize = 7#设置输出的图的图例字体大小
MyTitleSize = 14#设置输出的图的图例题目大小
Tune_month_OR_Pre_month = 0 # 0表示不进行Tune_month和Pre_month分析，修改cooperative_pro、num_cooperative
                            # 1表示Tune_month，
                            #2表示Pre_month，
                            # 3表示不同价格策略,此时num_cooperative为不同价格策略的方案；修改cooperative_pro都为一样、num_cooperative

Price_Change = [ ['pakchoi',[ [0,1]             ], [0]       ],

                 ['pakchoi',[ [30,40], [80,100] ], [0.8,0.8] ],
                 ['pakchoi',[ [30,40], [80,100] ], [0.5,0.5] ],
                 ['pakchoi',[ [30,40], [80,100] ], [0.15,0.15] ],

                 ['pakchoi',[ [30,40], [80,100] ], [-0.15,-0.15] ],
                 ['pakchoi',[ [30,40], [80,100] ], [-0.5,-0.5] ],
                 ['pakchoi',[ [30,40], [80,100] ], [-0.8,-0.8] ]  ]#几组策略，每组策略是品种，时间段，涨幅
Save_name = ""# 用来保存文件时区分文件夹
Order_Month  = 6#当YearOrMonthSupply为False时起作用。表示提前几个月下订单
# 最近12个月整个合作社的西红柿最低供给量为3.5吨/亩*6亩，黄瓜最低供给量为2吨/亩*10亩，菜花最低供给量为0.8吨/亩*10亩，生菜最低供给量为2吨/亩*10亩，#青椒和西红柿轮作最低供给量为2吨/亩*5亩
#action_name = ['0-nothing', '1-potato', '2-tomato', '3-cucumber','4-pakchoi','5-broccoli','6-cabbage','7-turnip','8lettuce',
# '9-wax_gourd','10-bean','11-pepper','12-eggplant','13-celery']
#[‘0-不种植’，'1-土豆', '2-西红柿', '3-黄瓜', '4-菠菜', '5-菜花', '6-洋白菜', '7-白萝卜', '8-生菜', '9-冬瓜', '10-豆角', '11-青椒', '12-茄子', '13-芹菜']
# 1, 2, 11, 12，'土豆', '西红柿','青椒', '茄子'轮作
#[4, 5, 6, 7]'菠菜', '菜花', '洋白菜', '白萝卜'轮作
#[3, 9]'黄瓜','冬瓜'轮作
#               0    1     2     3    4    5     6    7     8    9    10   11   12   13
Min_Demand =  [0.0, 0.0, 15.0, 20.0, 0.0,  5.0, 0.0, 0.0, 20.0, 0.0, 0.0, 10.0, 0.0, 0.0]
IMAGES_PATH = "./outputs_dynamic/images/"

Tune_month = [1] #调整频率，1表示每1个月调整一次
Pre_month = [1]#最小是1
# Tune_month = [100] #调整频率，1表示每1个月调整一次
# Pre_month = [20] #表示Pre_month个月的数据用来预训练，必须大于History_Month
# 建立从2012/1/1--2021/12/1对应的120个月份的列表
Month_List = []#第几个月对应的月份
Start_Month = datetime.date(2012,1,1)
print(Start_Month)
for i in range(Max_month):
 Month_List.append(Start_Month + relativedelta(months=i))

#价格预测相关参数
History_Month = 10 #需要几个月的数据预测下个月数据
Future_Month = 8 #价格预测时候，用来预测第几个月的价格,取生育期最长的蔬菜价格，目前是10月份种青椒，6月份收，间隔8个月

#和训练有关参数
Random_Seed = 25
Train_Eps = 200 # 训练的回合数
Max_Step = 120 #每回合迭代最大步长

Step_or_data = 0 #稳定步长还是随数据量积累而定？0表示最大步长，1表示随数据而定
#MyEnv Parameter
action_name = ['null', 'potato', 'tomato', 'cucumber','pakchoi','broccoli','cabbage','turnip','lettuce','wax_gourd','bean','pepper','eggplant','celery']
vegetable = ['potato', 'tomato', 'cucumber','pakchoi','broccoli','cabbage','turnip','lettuce','wax_gourd','bean','pepper','eggplant','celery']
vagetable_chinese_name = ['土豆','西红柿','黄瓜','菠菜', '菜花', '洋白菜', '白萝卜', '生菜', '冬瓜',  '豆角',  '青椒',  '茄子', '芹菜']
# 单位已转化为CNY/kg #WHY?表4-2，用上了，参考文献Designing price-contingent vegetable rotation schedules__using agent-based simulation
Plant_Cost = [1, 1.72, 1.63, 1.29, 1.5, 0.67, 0.42, 1.25, 1.07, 2.94, 1.94, 1.71, 0.86]#13位 和env中的self.plant_cost一致


#产量转换为吨/亩，也就是10^3公斤/亩
yeild = [[[0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # 动作为空
                      [[0, 0, 0, 1.75, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 2.25, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.75, -1],
                       [-1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.75],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]],  # potato


                       [[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2.5, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 2, 2.25, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2.083, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1.5, 2.25, -1, 0, 0, 0, 0, 0]],  # tomato


                       [[0, 0, 0, 2, 2.2, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 1.5, -1, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2.5, -1, 0, 0, 0, 0, 0, 0]],  # cucumber修改前的
                      [[0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0.8, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0.8, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2],
                       [0, 1.2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0]],  # pakchoi
                      [[0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0.8, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0.8, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.25, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                       [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0.8, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0.8, -1, 0, 0, 0, 0, 0, 0]],  # broccoli
                      [[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 2, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 2, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 2, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1, 0],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2.5, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 4.5, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]],  # cabbage
                      [[0, 0, 2.75, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 2.75, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 3.5, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3.5, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 3.5, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 3.5, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.5, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                       [0, 2.75, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 2.75, -1, 0, 0, 0, 0, 0, 0, 0, 0]],  # turnip
                      [[0, 0, 0, 0, 3.25, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 2, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 2, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 2, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 2, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 3.25, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.25, -1],
                       [3.25, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 3.25, -1, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                       [0, 0, 0, 3.25, -1, 0, 0, 0, 0, 0, 0, 0]],  # lettuce
                      [[0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 4.5, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 5, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]],  # Chinese_watermelon

                       [[0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1.4, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1.25, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1.75, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1.2, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1.5, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1.283, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],  # green_bean

                      [[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                      [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 3, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 2.75, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0]],  # green_pepper

                      [[0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 3.25, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 2.5, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]],  # eggplant
                      [[0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 4.25, -1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 2.75, -1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 2.75, -1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.5, -1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.5, -1],
                       [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]]   # celery
                      ]