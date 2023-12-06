import csv
import pandas as pd
import datetime
import pickle
from pandas.core.frame import DataFrame
from matplotlib import pyplot as plt
import matplotlib as mpl
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import train_test_split
from sklearn import ensemble#导入集成学习库
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import glob
import numpy as np
import Parameter
import time
from dateutil.relativedelta import relativedelta
import simulation_plot
from common.utils import make_dir
import os
import random

def read_data(filename):
	data = []
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		#print(type(reader))
		for row in reader:
			data.append([row[0],float(row[2])])
				
	data_month = {}
	price_raw = []
	sum_week = 0
	for i in range(len(data)):
		date = data[i][0]
		data_month[date] = data[i][1]


	m = pd.DataFrame.from_dict(data_month,orient = 'index',columns = ['price'])

	m.index = pd.to_datetime(m.index)  # 将字符串索引转换成时间索引

	decomposition = seasonal_decompose(m)
	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid
	trend = trend.dropna()
	#print(trend)
	residual = residual.dropna()
	#print(residual)
	return m,trend,seasonal,residual


def read_data_change(filename,vegetable, num_cooperate):
	data = []
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		# print(type(reader))
		for row in reader:
			data.append([row[0], float(row[2])])

	data_month = {}
	price_raw = []
	sum_week = 0
	for i in range(len(data)):
		date = data[i][0]
		data_month[date] = data[i][1]

	if Parameter.Tune_month_OR_Pre_month == 3:  # 价格变化
		# 价格调整
		if vegetable == Parameter.Price_Change[num_cooperate][0]:
			for m in range(len(Parameter.Price_Change[num_cooperate][1])):
				start = Parameter.Price_Change[num_cooperate][1][m][0]
				end = Parameter.Price_Change[num_cooperate][1][m][1] + 1
				l = len(data)
				if l < start:
					break
				elif l >= start and l < end:
					end = l + 1

				for n in range(start, end):
					part_date = data[n][0]
					data_month[part_date] = data[n][1] * (Parameter.Price_Change[num_cooperate][2][m] + 1)

	m = pd.DataFrame.from_dict(data_month, orient='index', columns=['price'])

	# m = pd.DataFrame.from_dict(data_week,orient = 'index')
	m.index = pd.to_datetime(m.index)  # 将字符串索引转换成时间索引

	decomposition = seasonal_decompose(m)
	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid
	trend = trend.dropna()
	# print(trend)
	residual = residual.dropna()
	# print(residual)
	return m, trend, seasonal, residual

def getRandomIndex(n, x):
    # 索引范围为[0, n), 随机选x个不重复
    index = random.sample(range(n), x)
    return index

def data_split_1(m,history_month):#根据history_month个月的价格预测下个月的
	#print(len(m))
	price_list = []
	for i in range(len(m)):
		price_list.append(m.iloc[i].price)#即对数据进行位置索引，从而在数据表中提取出相应的数据
	#print(price_list)
	X = []
	Y = []
	X_train = []
	y_train = []
	X_test = []
	y_test = []

	for i in range(len(price_list)-history_month):
		X.append(price_list[i:i+history_month])
		Y.append(price_list[i+history_month])
	#print('the number of data for GBDT is:',len(X_week))

	data_num = len(X)
	data_split = int(data_num*0.9)
	# 顺序划分训练集和测试集
	# X_train = X[0:data_split]
	# y_train = Y[0:data_split]
	# X_test = X[data_split:]
	# y_test = Y[data_split:]

	#随机选取训练集和测试集
	# 先根据上面的函数获取训练集索引train_index
	train_index = np.array(getRandomIndex(data_num, data_split))
	#print("train_index", train_index)
	# 再将train_index从总的index中减去就得到了训练集索引test_index
	test_index = np.delete(np.arange(data_num), train_index)
	#print("test_index", test_index)
	for i in train_index:
		X_train.append(X[i])
		y_train.append(Y[i])
	for i in test_index:
		X_test.append(X[i])
		y_test.append(Y[i])

	return X_train,y_train,X_test,y_test
def data_split_AccordingMixedData(data_list, end_month_list,history_month):#根据history_month个月的价格预测下个月的

	X = []
	Y = []
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	for v in range(len(data_list)):
		m = data_list[v]['2012-1-1':end_month_list[v]]
		price_list = []
		for i in range(len(m)):
			price_list.append(m.iloc[i].price)#即对数据进行位置索引，从而在数据表中提取出相应的数据
		for i in range(len(price_list)-history_month):
			X.append(price_list[i:i+history_month])
			Y.append(price_list[i+history_month])

	data_num = len(X)
	data_split = int(data_num*0.9)
	#随机选取训练集和测试集
	# 先根据上面的函数获取训练集索引train_index
	train_index = np.array(getRandomIndex(data_num, data_split))
	#print("train_index", train_index)
	# 再将train_index从总的index中减去就得到了训练集索引test_index
	test_index = np.delete(np.arange(data_num), train_index)
	#print("test_index", test_index)
	for i in train_index:
		X_train.append(X[i])
		y_train.append(Y[i])
	for i in test_index:
		X_test.append(X[i])
		y_test.append(Y[i])

	return X_train,y_train,X_test,y_test


#a是预测值，b是实际值
def mrae_cal(a,b):
	mrae = []
	for i in range(len(a)):
		if a[i] > b[i]:
			error = a[i] - b[i]
		else:
			error = b[i] - a[i]
		mrae.append((error/b[i])*100)
	return sum(mrae)/len(mrae)


def GBDT_algorithm(parameter,X_train,y_train,X_test,y_test):
	clf = ensemble.GradientBoostingRegressor(**parameter)
	# 算法中选用的损失函数
	# 对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”
	# 默认是均方差"ls"
	# 一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好
	# 如果是噪音点较多，推荐用抗噪音的损失函数"huber"
	# 而如果我们需要对训练集进行分段预测的时候，则采用“quantile”
	# https://www.freesion.com/article/3028535419/
	# 估计器拟合训练数据
	clf.fit(X_train, y_train)
	# 训练完的估计器对测试数据进行预测
	y_pred = clf.predict(X_test)
	mae = mean_absolute_error(y_test, y_pred)#平均绝对误差（Mean Absolute Error，MAE）
	#print('the mean_absolute_error of the average week price is:%.2f'%mae)
	mrae = mrae_cal(y_pred,y_test)#平均相对误差
	mape = mean_absolute_percentage_error(y_test,y_pred)#平均绝对百分比误差 ,mrae / 100 = mape
	#print('the mean relative error of the average week price is:%.2f'%(mrae),'%')
	#print('————————————————————————————————————————————————————————')
	return clf,y_pred,mae,mrae,mape



vegetable = ['potato', 'tomato', 'cucumber','pakchoi','broccoli','cabbage','turnip','lettuce','wax_gourd','bean','pepper','eggplant','celery']


plot_1 = []
plot_2 = []
plot_3 = []
plot_4 = []
plot_5 = []

mae_diff = []
letter = list(map(chr, range(ord('A'), ord('Z') + 1)))
Test = 0

def TrainPricePredict_GBDT(history_month,address): #根据history_month个月的价格预测下个月的
	for i in range(len(vegetable)):
		if 	Test == 0:#外部文件调用
			filename = 'price_data/' + str(address)+ '月均价格-'+ vegetable[i] +'.csv'
		else:#内测
			filename = str(address)+ '月均价格-' + vegetable[i] + '.csv'
		m,trend,seasonal,residual = read_data(filename)

		#print(m)
		X_train_1,y_train_1,X_test_1,y_test_1 = data_split_1(m['2012-1-1':'2021-12-1'],history_month)

		print(len(X_train_1))
		print(len(X_test_1))

		params_1 = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 3,
					 #'learning_rate': 0.05, 'loss': 'lad'}
					 'learning_rate': 0.05, 'loss': 'absolute_error', 'random_state': 2 }
		#random_state：int RandomState 实例或无，默认=无;控制在每次提升迭代中给予每个树估计器的随机种子。此外，它还控制每次拆分时特征的随机排列;Popular integer random seeds are 0 and 42

		clf_1,y_pred_1, mae_1,mrae_1,mape_1 = GBDT_algorithm(params_1,X_train_1,y_train_1,X_test_1,y_test_1)
		if 	Test == 0:
			savename = 'price_data/'  + str(address)+ 'clf_1' + vegetable[i] + '.pickle'
		else:
			savename =  str(address) + 'clf_1' + vegetable[i] + '.pickle'
		with open(savename, 'wb') as f:
			pickle.dump(clf_1, f)


def SingleTrain(m,end_month,history_month):#根据history_month个月的价格预测下个月的
	X_train_1, y_train_1, X_test_1, y_test_1 = data_split_1(m['2012-1-1':end_month], history_month)

	print(len(X_train_1))
	print(len(X_test_1))

	params_1 = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 3,
				# 'learning_rate': 0.05, 'loss': 'lad'}
				'learning_rate': 0.05, 'loss': 'absolute_error', 'random_state': 2 }
		#random_state：int RandomState 实例或无，默认=无;控制在每次提升迭代中给予每个树估计器的随机种子。此外，它还控制每次拆分时特征的随机排列;Popular integer random seeds are 0 and 42


	clf_1, y_pred_1, mae_1, mrae_1, mape_1 = GBDT_algorithm(params_1, X_train_1, y_train_1, X_test_1, y_test_1)
	return clf_1, y_pred_1, mae_1, mrae_1, mape_1


def SingleTrain_AccodingMixedData(data_list, end_month_list, history_month):  # 根据history_month个月的价格预测下个月的

	X_train_1, y_train_1, X_test_1, y_test_1 = data_split_AccordingMixedData(data_list, end_month_list, history_month)

	print(len(X_train_1))
	print(len(X_test_1))

	params_1 = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 3,
				# 'learning_rate': 0.05, 'loss': 'lad'}
				'learning_rate': 0.05, 'loss': 'absolute_error','random_state': 2 }
		#random_state：int RandomState 实例或无，默认=无;控制在每次提升迭代中给予每个树估计器的随机种子。此外，它还控制每次拆分时特征的随机排列;Popular integer random seeds are 0 and 42


	clf_1, y_pred_1, mae_1, mrae_1, mape_1 = GBDT_algorithm(params_1, X_train_1, y_train_1, X_test_1, y_test_1)
	return clf_1, y_pred_1, mae_1, mrae_1, mape_1



def TrainPricePredict_GBDT_DynamicTest(history_month,TruePrice_month, address): #根据history_month个月的价格预测下个月的,
	# 测试一下随着数据量的增多对精度的影响变化
	# TruePrice_month为用多少个月的数据开始训练
	mae_1_list = [[] for i in range(Parameter.num_crop)]
	mrae_1_list = [[] for i in range(Parameter.num_crop)]
	mape_1_list = [[] for i in range(Parameter.num_crop)]
	#price_crop_list = [[] for i in range(Parameter.num_crop)]

	for i in range(len(vegetable)):

		filename = str(address) + '月均价格-' + vegetable[i] + '.csv'
		m,trend,seasonal,residual = read_data(filename)

		for j in range (len(m) - TruePrice_month - 1):

			#X_train_1,y_train_1,X_test_1,y_test_1 = data_split_1(m['2012-1-1':'2021-12-1'],history_month)
			end_month = str(Parameter.Start_Month + relativedelta(months=j + TruePrice_month + 1))
			clf_1, y_pred_1, mae_1, mrae_1, mape_1 = SingleTrain(m,end_month,history_month)
			#savename = 'price_data/clf_1' + vegetable[i] + '.pickle'
			savename = str(address) + 'clf_1' + vegetable[i] + '.pickle'
			with open(savename, 'wb') as f:
				pickle.dump(clf_1, f)

			mae_1_list[i].append(mae_1)
			mrae_1_list[i].append(mrae_1)
			mape_1_list[i].append(mape_1)

	return mae_1_list, mrae_1_list, mape_1_list

def TrainPricePredict_GBDT_AccodingData(history_month,accumulative_month,address): #根据history_month个月的价格预测下个月的(history_month): #根据history_month个月的价格预测下个月的,测试一下随着数据量的增多对精度的影响变化

	for l in range(len(address)):
		for i in range(len(vegetable)):
			#filename = 'price_data/月均价格-'+ vegetable[i] +'.csv'

			filename = str(address[l]) + '月均价格-' + vegetable[i] + '.csv'
			m,trend,seasonal,residual = read_data(filename)
			# X_train_1,y_train_1,X_test_1,y_test_1 = data_split_1(m['2012-1-1':'2021-12-1'],history_month)
			end_month = str(Parameter.Start_Month + relativedelta(months= accumulative_month + 1))
			clf_1, y_pred_1, mae_1, mrae_1, mape_1 = SingleTrain(m, end_month, history_month)
			# savename = 'price_data/clf_1' + vegetable[i] + '.pickle'
			savename = str(address[l]) + 'clf_1' + vegetable[i] + '.pickle'
			with open(savename, 'wb') as f:
				pickle.dump(clf_1, f)

def TrainPricePredict_GBDT_AccodingMixedData(history_month,accumulative_month,num_cooperate, address): #根据history_month个月的价格预测下个月的(history_month): #根据history_month个月的价格预测下个月的,测试一下随着数据量的增多对精度的影响变化

	for l in range(len(address)):
		for i in range(len(vegetable)):
			data_list = []
			end_month_list = []
			for j in range(len(address[l])):
				filename = str(address[l][j]) + '月均价格-' + vegetable[i] + '.csv'
				m,trend,seasonal,residual = read_data_change(filename, vegetable[i], num_cooperate)
				data_list.append(m)
				end_month = str(Parameter.Start_Month + relativedelta(months= accumulative_month[j] + 1))
				end_month_list.append(end_month)
			clf_1, y_pred_1, mae_1, mrae_1, mape_1 = SingleTrain_AccodingMixedData(data_list, end_month_list, history_month)
			# savename = 'price_data/clf_1' + vegetable[i] + '.pickle'
			if len(address[l]) == 1:
				savename = str(address[l][j]) + 'clf_1' + vegetable[i] + '.pickle'
			else:
				new_address = '混合数据模型/'
				make_dir(str(new_address))
				savename = str(new_address) + 'clf_1' + vegetable[i] + '.pickle'
			with open(savename, 'wb') as f:
				pickle.dump(clf_1, f)

def PricePredict_GBDT(crop_name,X_list,model_address):
	predict_result = []
	if 	Test == 0:
		model_filename = glob.glob('price_data/' + str(model_address) +'*' + crop_name + '.pickle')
	else:
		model_filename = glob.glob(str(model_address) +'*' + crop_name + '.pickle')
	pickle_in = open(model_filename[0], 'rb')
	clf = pickle.load(pickle_in)

	X_predict = np.array(X_list).reshape(1, -1)
	predict_result = clf.predict(X_predict)
	return predict_result[0]
def Init_Price(address):
	price = []

	from os.path import dirname, abspath
	for i in range(Parameter.num_crop):
		filename = dirname(abspath(__file__)).replace("environment", "") + '/'+ str(address) +  '月均价格-' + \
				   Parameter.action_name[i + 1] + '.csv'
		# filename = r'../price_data/月均价格-'+self.action_name[i+1]+'.csv'
		with open(filename, 'r') as csvfile:
			reader = csv.reader(csvfile)
			price.append([float(row[2]) for row in reader])

	return price
def Draw_Init_Price(address,label):
	label_1 = []
	price_crop = [[] for i in range(Parameter.num_crop)]
	for l in range(len(address)):
		price_init = Init_Price(address[l])
		for i in range(Parameter.num_crop):
			price_crop[i].append(price_init[i])
		label_1.append(str(label[l]))


	# 画图和保存数据
	x = [v for v in range(len(price_crop[0][0]))]
	for i in range(Parameter.num_crop):
		# Where to save the figures
		IMAGES_PATH = "./Init_Price/images"
		os.makedirs(IMAGES_PATH, exist_ok=True)

		plt.figure()
		# plt.title('Month Price')
		plt.tight_layout()

		simulation_plot.simulation_plot_9(x, price_crop[i], label_1, Parameter.vegetable[i])
		plt.xlabel("month")
		plt.ylabel("price")
		plt.legend()
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_price")

		make_dir('./Init_Price/')
		filename = './Init_Price/' + str(Parameter.vegetable[i]) + '_price_test4.csv'
		for j in range(len(price_crop[i][0])):
			p = []
			for m in range(len(price_crop[i])):
				p.append(price_crop[i][m][j])
			p.insert(0, x[j])
			SaveData(p, filename, ['x'] + label_1, j)

def Update_PriceList_ForTest(TruePrice_month, pre_month, data_address, model_address):
	time_start = time.perf_counter()  # 记录开始时间
	print('Update_PriceList start!')
	price = []
	from os.path import dirname, abspath
	for i in range(Parameter.num_crop):
		filename = dirname(abspath(__file__)).replace("environment", "") + '/'+ str(data_address) + '月均价格-' + \
				   Parameter.action_name[i + 1] + '.csv'
		# filename = r'../price_data/月均价格-'+self.action_name[i+1]+'.csv'
		with open(filename, 'r') as csvfile:
			reader = csv.reader(csvfile)
			price.append([float(row[2]) for row in reader])

		# 只读当前月份Num_Month之前的数据进行训练

		l = len(price[i])
		del price[i][TruePrice_month:]
		print("使用了 %d 个月的真实数据" % TruePrice_month)
		if TruePrice_month < Parameter.History_Month:
			print("价格数据量不足，不能预测未来数据, 需要 %d 个月的数据" % Parameter.History_Month)
			for j in range(Parameter.Future_Month + 1 - TruePrice_month):  # 根据预测算法代替真实价格数据,需要未来Future_Month个数据
				price[i].append(0.0)
		else:
			max_ = max(Parameter.Future_Month, pre_month)  # 用2者最大预测
			print("预测并使用了 %d 个月的数据" % (max_))
			for j in range(max_):  # 根据预测算法代替真实价格数据
				# price[i].append(
				# 	(PricePredict_GBDT(Parameter.vegetable[i], price[i][-Parameter.History_Month:])))
				res = PricePredict_GBDT(Parameter.vegetable[i], price[i][-Parameter.History_Month:],model_address)
				# res_2 = price[i][-1]
				# if abs(res - res_2) <= 0.001:
				# 	mmm = 0
				price[i].append(res)

	time_end = time.perf_counter()  # 记录开始时间
	time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

	print('Update_PriceList end!')
	print('Update_PriceList运行时间为', time_sum)

	return price

# TruePrice_month表示需要多少个月的真实数据；pre_month表示需要预测几个月的数据，注意和Parameter.Future_Month二者取最大
def Update_PriceList_Change_ForTest(TruePrice_month, pre_month, Price_Change):
	time_start = time.perf_counter()  # 记录开始时间
	print('Update_PriceList start!')
	price = []

	from os.path import dirname, abspath
	for i in range(Parameter.num_crop):
		filename = dirname(abspath(__file__)).replace("environment", "") + '/宁波数据/月均价格-' + \
				   Parameter.action_name[i + 1] + '.csv'
		# filename = r'../price_data/月均价格-'+self.action_name[i+1]+'.csv'
		with open(filename, 'r') as csvfile:
			reader = csv.reader(csvfile)
			price.append([float(row[2]) for row in reader])

		# 价格调整
		if Parameter.vegetable[i] == Price_Change[0]:
			for m in range(len(Price_Change[1])):
				start = Price_Change[1][m][0]
				end = Price_Change[1][m][1] + 1
				l = len(price[i])
				if l < start:
					break
				elif l >= start and l < end:
					end = l + 1
				part = price[i][start:end]
				price[i][start: end] = [v * (Price_Change[2][m] + 1) for v in part]

		# 只读当前月份Num_Month之前的数据进行训练
		del price[i][TruePrice_month:]
		print("使用了 %d 个月的真实数据" % TruePrice_month)
		if TruePrice_month < Parameter.History_Month:
			print("价格数据量不足，不能预测未来数据, 需要 %d 个月的数据" % Parameter.History_Month)
			for j in range(Parameter.Future_Month + 1 - TruePrice_month):  # 根据预测算法代替真实价格数据,需要未来Future_Month个数据
				price[i].append(0.0)
		else:
			max_ = max(Parameter.Future_Month, pre_month)  # 用2者最大预测
			print("预测并使用了 %d 个月的数据" % (max_))
			model_address = '宁波数据/'
			for j in range(max_):  # 根据预测算法代替真实价格数据
				price[i].append(
					(PricePredict_GBDT(Parameter.vegetable[i], price[i][-Parameter.History_Month:],model_address)))

	time_end = time.perf_counter()  # 记录开始时间
	time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

	print('Update_PriceList end!')
	print('Update_PriceList运行时间为', time_sum)

	return price
def SaveData(data,filename,str_col,times):

    result = pd.DataFrame([data], columns=str_col)

    if times == 0:
        result.to_csv(filename, mode='w', index=False, encoding='gbk')  # header参数为None表示不显示表头
    else:
        result.to_csv(filename, mode='a', index=False, encoding='gbk', header=False)  # header参数为None表示不显示表头


def Save_Fig(images_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
	# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
	# 指定dpi=200，图片尺寸为 1200*800
	# 指定dpi=300，图片尺寸为 1800*1200
    #标签label等属性设置
    mpl.rc('axes', labelsize=20)
    mpl.rc('xtick', labelsize=20)
    mpl.rc('ytick', labelsize=20)
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    #轴标签、标题、刻度标签等等会超出图形区域，导致显示不全
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, bbox_inches='tight')



def PricePredectTest_0(start_month_predict,data_address, model_address,label_reg):##测试一下不停的预测结果,价格有变化
	price_list = []
	label_ = []
	price_crop = [[] for i in range(Parameter.num_crop)]
	for l in range(len(model_address)):#不同模型的预测结果
		for i in range(Parameter.num_cooperative):
			if Parameter.Tune_month_OR_Pre_month == 3:
				# price_list.append(Update_PriceList_Change_ForTest(TruePrice_month, Parameter.Max_Step-TruePrice_month+Parameter.Future_Month, Parameter.Price_Change[i]))
				price_list.append(Update_PriceList_Change_ForTest(start_month_predict,
																  Parameter.Max_Step - start_month_predict,
																  Parameter.Price_Change[i]))
			else:
				# price_list.append(Update_PriceList_ForTest(TruePrice_month, Parameter.Max_Step - TruePrice_month + Parameter.Future_Month))
				price_list.append(Update_PriceList_ForTest(start_month_predict,
														   Parameter.Max_Step - start_month_predict,data_address[l], model_address[l]))

	for i in range(Parameter.num_crop):
		price_ = []
		#for j in range(Parameter.num_cooperative):
		for j in range(len(price_list)):

			if Parameter.Tune_month_OR_Pre_month == 3 and len(model_address) == 1:#多价格比较
				if Parameter.vegetable[i] == Parameter.Price_Change[j][0]:
					price_.append(price_list[j][i])
				else:
					if j == 0:
						price_.append(price_list[j][i])
				price_crop[i] = price_



			elif len(model_address) > 1 :#其它情况比如多模型的结果比较
				price_.append(price_list[j][i])
				price_crop[i].append(price_[-1])

			else:
				print('设置错误!')
				assert(0)

	address = '宁波数据/'
	price_init = Init_Price(address)
	for i in range(Parameter.num_crop):
		price_crop[i].insert(0, price_init[i])

	if Parameter.Tune_month_OR_Pre_month == 3 and len(model_address) == 1:  # 多价格比较
		for p in range(len(Parameter.Price_Change) + 1):  # 多价格比较
			if p == 0:
				label_.append('price initial ')
			else:
				label_.append('price Predict_GBDT_' + str(p))
	elif len(model_address) > 1:  # 其它情况比如多模型的结果比较
		for l in range(len(model_address)):  # 其它情况比如多模型的结果比较
			label_.append('price Predict_GBDT_' + str(label_reg[l]))
		label_.insert(0, 'price initial ')
	else:
		print('设置错误!')
		assert (0)

	x = [v for v in range(len(price_crop[0][0]))]
	for i in range(Parameter.num_crop):
		# Where to save the figures
		IMAGES_PATH = "./Test_0/images"
		os.makedirs(IMAGES_PATH, exist_ok=True)

		plt.figure()
		# plt.title('Month Price')
		plt.tight_layout()

		simulation_plot.simulation_plot_9(x, price_crop[i], label_, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Price(Yuan)")
		count = 0
		for p in range(len(Parameter.Price_Change)):
			if Parameter.vegetable[i] == Parameter.Price_Change[p][0]:
				count += 1
		if count > 0:
			plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={'size': 7})
		else:
			plt.legend()

		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "price")

		make_dir('./Test_0/')
		filename = './Test_0/' + str(Parameter.vegetable[i]) + '_price_test0.csv'
		for j in range(len(price_crop[i][0])):
			data_ = []
			data_.append(x[j])
			for m in range(len(price_crop[i])):
				data_.append(price_crop[i][m][j])
			SaveData(data_, filename, ['x'] + label_, j)


def PricePredectTest_1(history_month_list, TruePrice_month):#1.随着数据量的增多，mape等指标变化，图和数据保存;；预测模型参数history_month更换
	# TruePrice_month为用多少个月的数据开始训练
	address = '宁波数据/'
	#address = '新发地数据/'
	mae_1_list = [[] for i in range(Parameter.num_crop)]
	mrae_1_list = [[] for i in range(Parameter.num_crop)]
	mape_1_list = [[] for i in range(Parameter.num_crop)]
	#price_crop_list = [[] for i in range(Parameter.num_crop)]
	label_1 = []
	label_2 = []
	#label_3 = []
	for m in range(len(history_month_list)):
		if history_month_list[m] > TruePrice_month:
			print('历史数据不足！')
			assert (0)
		mae_1_list_res, mrae_1_list_res, mape_1_list_res = TrainPricePredict_GBDT_DynamicTest(history_month_list[m],TruePrice_month,address)
		for i in range(Parameter.num_crop):
			mae_1_list[i].append(mae_1_list_res[i])
			mape_1_list[i].append(mape_1_list_res[i])
			#price_crop_list[i].append(price_crop_list_res[i])
		label_1.append('mae_' + str(history_month_list[m]))
		label_2.append('mape_' + str(history_month_list[m]))
		#label_3.append('price_' + str(history_month_list[m]))

	for i in range(Parameter.num_crop):
		# Where to save the figures
		IMAGES_PATH = "./Test_1/images"
		os.makedirs(IMAGES_PATH, exist_ok=True)
		plt.figure()
		# plt.title('Month Price')
		plt.tight_layout()
		x = [v + TruePrice_month + 1 for v in range(len(mae_1_list[i][0]))]
		simulation_plot.simulation_plot_9(x, mae_1_list[i], label_1, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mae")
		plt.legend()
		# Where to save the figures
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mae")

		plt.figure()
		simulation_plot.simulation_plot_9(x, mape_1_list[i], label_2, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mape")
		plt.legend()
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mape")

		make_dir('./Test_1/')

		filename1 = './Test_1/' + str(Parameter.vegetable[i]) + '_mae_test1.csv'
		filename2 = './Test_1/' + str(Parameter.vegetable[i]) + '_mape_test1.csv'


		for j in range(len(mae_1_list[i][0])):
			mae = []
			mape = []

			for m in range(len(mae_1_list[i])):
				mae.append(mae_1_list[i][m][j])
				mape.append(mape_1_list[i][m][j])

			mae.insert(0, x[j])
			mape.insert(0, x[j])

			SaveData(mae, filename1, ['x'] +label_1, j)
			SaveData(mape, filename2, ['x'] + label_2, j)



def PricePredectTest_2(history_month, TruePrice_month):#2.对着某一个预测模型，多少个周期后会变的慢慢不准了.
	#随着数据量的增多得到的预测模型用来不停的预测的时候，多少个周期会慢慢变的不准了
	# history_month为训练用参数，根据history_month个月数据预测下个月数据
	# TruePrice_month为用多少个月的数据开始训练
	price_list = []
	label_1 = []
	label_2 = []
	label_3 = []

	price_crop_mae = [[] for i in range(Parameter.num_crop)]
	price_crop_mape = [[] for i in range(Parameter.num_crop)]
	price_crop = [[] for i in range(Parameter.num_crop)]

	address = '宁波数据/'
	price_init = Init_Price(address)
	address = ['宁波数据/']
	data_address = '宁波数据/'
	model_address = '宁波数据/'
	for m in range(len(TruePrice_month)):
		#训练算法
		TrainPricePredict_GBDT_AccodingData(history_month, TruePrice_month[m],address)

		for i in range(Parameter.num_cooperative):
			price_list.append(Update_PriceList_ForTest(TruePrice_month[m],Parameter.Max_Step - TruePrice_month[m],data_address,model_address))

		for i in range(Parameter.num_crop):
			price_ = []

			price_.append(price_list[m][i])

			err = []
			perc_err = []
			for j in range(len(price_init[i])):
				err.append(mean_absolute_error([ price_init[i][j] ], [ price_[0][j] ]))# 平均绝对误差（Mean Absolute Error，MAE）
				perc_err.append(mean_absolute_percentage_error([ price_init[i][j] ], [ price_[0][j] ])) # 平均绝对百分比误差 ,mrae / 100 = mape
			price_crop_mae[i].append(err)
			price_crop_mape[i].append(perc_err)
			price_crop[i].append(price_[-1])
		label_1.append('mae_' + str(TruePrice_month[m]))
		label_2.append('mape_' + str(TruePrice_month[m]))
		label_3.append('price_' + str(TruePrice_month[m]))

	address = '宁波数据/'
	price_init = Init_Price(address)
	for i in range(Parameter.num_crop):
		price_crop[i].insert(0, price_init[i])
	label_3.insert(0, 'price initial_NingBo')

	x = [v for v in range(len(price_crop_mae[0][0]))]
	for i in range(Parameter.num_crop):
		# Where to save the figures
		IMAGES_PATH = "./Test_2/images"
		os.makedirs(IMAGES_PATH, exist_ok=True)

		plt.figure()
		# plt.title('Month Price')
		plt.tight_layout()

		simulation_plot.simulation_plot_9(x, price_crop_mae[i], label_1, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mae")
		plt.legend()
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mae")

		plt.figure()
		simulation_plot.simulation_plot_9(x, price_crop_mape[i], label_2, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mape")
		plt.legend()
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mape")

		plt.figure()
		simulation_plot.simulation_plot_9(x, price_crop[i], label_3, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Price")
		plt.legend()
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_price")

		make_dir('./Test_2/')
		filename1 = './Test_2/' + str(Parameter.vegetable[i]) + '_mae_test2.csv'
		filename2 = './Test_2/' + str(Parameter.vegetable[i]) + '_mape_test2.csv'
		filename3 = './Test_2/' + str(Parameter.vegetable[i]) + '_price_test2.csv'

		for j in range(len(price_crop_mae[i][0])):
			mae = []
			mape = []
			p = []
			for m in range(len(TruePrice_month)):
				mae.append(price_crop_mae[i][m][j])
				mape.append(price_crop_mape[i][m][j])

			for m in range(len(price_crop[i])):
				p.append(price_crop[i][m][j])

			mae.insert(0,x[j])
			mape.insert(0,x[j])
			p.insert(0, x[j])

			SaveData(mae, filename1, ['x']+label_1, j)
			SaveData(mape, filename2, ['x']+label_2, j)
			SaveData(p, filename3, ['x']+label_3, j)


def PricePredectTest_3(TruePrice_month):#4.北京的预测模型，用来预测宁波的会有指标上的大的差异吗
	# TruePrice_month为用多少个月的数据开始训练
	address = '宁波数据/'
	mae_1_list, mrae_1_list, mape_1_list = TrainPricePredict_GBDT_DynamicTest(Parameter.History_Month,TruePrice_month,address)
	address = '新发地数据/'
	mae_2_list, mrae_2_list, mape_2_list = TrainPricePredict_GBDT_DynamicTest(Parameter.History_Month,TruePrice_month, address)


	for i in range(Parameter.num_crop):
		# Where to save the figures
		IMAGES_PATH = "./Test_3/images"
		os.makedirs(IMAGES_PATH, exist_ok=True)

		plt.figure()
		# plt.title('Month Price')
		plt.tight_layout()
		x = [v + TruePrice_month + 1 for v in range(len(mae_1_list[i]))]
		label_ = ['mae_NingBo','mae_BeiJing']
		simulation_plot.simulation_plot_9(x, [mae_1_list[i],mae_2_list[i]], label_, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mae")
		plt.legend()
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mae")

		plt.figure()
		label_ = ['mape_NingBo','mape_BeiJing']
		simulation_plot.simulation_plot_9(x, [mape_1_list[i],mape_2_list[i]], label_, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mape")
		plt.legend()
		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mape")

		make_dir('./Test_3/')
		filename = './Test_3/' + str(Parameter.vegetable[i]) + '_Test_3.csv'
		for j in range(len(mae_1_list[i])):
			SaveData([x[j], mae_1_list[i][j], mape_1_list[i][j], mae_2_list[i][j], mape_2_list[i][j]], filename, ['x', 'mae_NingBo', 'mape_NingBo', 'mae_BeiJing', 'mape_BeiJing'], j)

def RegionalDataPrediction(price_init,history_month, start_month_predict, TruePrice_month, TrainData_address, UseData_address,
									 model_address):
	price_list = []
	price_crop_mae = [[] for i in range(Parameter.num_crop)]
	price_crop_mape = [[] for i in range(Parameter.num_crop)]
	price_crop = [[] for i in range(Parameter.num_crop)]
	# 训练算法
	TrainPricePredict_GBDT_AccodingData(history_month, TruePrice_month, TrainData_address)

	for i in range(Parameter.num_cooperative):
		price_list.append(
			Update_PriceList_ForTest(start_month_predict, Parameter.Max_Step - start_month_predict, UseData_address,
									 model_address))

	for i in range(Parameter.num_crop):
		price_ = []
		price_.append(price_list[0][i])
		err = []
		perc_err = []

		for j in range(len(price_init[i])):
			err.append(mean_absolute_error([price_init[i][j]], [price_[0][j]]))  # 平均绝对误差（Mean Absolute Error，MAE）
			perc_err.append(
				mean_absolute_percentage_error([price_init[i][j]], [price_[0][j]]))  # 平均绝对百分比误差 ,mrae / 100 = mape


		price_crop_mae[i].append(err)
		price_crop_mape[i].append(perc_err)
		price_crop[i].append(price_)

	return price_crop_mae, price_crop_mape, price_crop

def MixedRegionalDataPrediction(price_init,history_month, start_month_predict, TruePrice_month, TrainData_address, UseData_address,
									 model_address):#混合数据预测
	price_list = []
	price_crop_mae = [[] for i in range(Parameter.num_crop)]
	price_crop_mape = [[] for i in range(Parameter.num_crop)]
	price_crop = [[] for i in range(Parameter.num_crop)]
	# 训练算法
	TrainPricePredict_GBDT_AccodingMixedData(history_month, TruePrice_month, 0, TrainData_address)

	for i in range(Parameter.num_cooperative):
		price_list.append(
			Update_PriceList_ForTest(start_month_predict, Parameter.Max_Step - start_month_predict, UseData_address,
									 model_address))

	for i in range(Parameter.num_crop):
		price_ = []
		price_.append(price_list[0][i])
		err = []
		perc_err = []

		for j in range(len(price_init[i])):
			err.append(mean_absolute_error([price_init[i][j]], [price_[0][j]]))  # 平均绝对误差（Mean Absolute Error，MAE）
			perc_err.append(
				mean_absolute_percentage_error([price_init[i][j]], [price_[0][j]]))  # 平均绝对百分比误差 ,mrae / 100 = mape


		price_crop_mae[i].append(err)
		price_crop_mape[i].append(perc_err)
		price_crop[i].append(price_)

	return price_crop_mae, price_crop_mape, price_crop

def PricePredectTest_4(history_month, start_month_predict, TruePrice_month,compare_address, UseData_address, Traindata_address, model_address,label):#北京的预测模型，用来预测宁波的会有指标上的大的差异吗（实验二）;北京的预测模型，用来预测宁波的数据结果
	# history_month为训练用参数，根据history_month个月数据预测下个月数据
	# start_month_predict为根据TruePrice_month月的数据训练好模型后，从start_month_predict个月开始预测模型
	# TruePrice_month为用多少个月的数据开始训练
	# compare_address为对比组原始数据；UseData_address为使用哪里的数据开始预测
	# 北京和宁波的模型训练，对同一组宁波数据进行预测；Traindata_address为训练数据
	# model_address为采用哪里的模型
	label_1 = []
	label_2 = []
	label_3 = []
	price_crop_mae = [[] for i in range(Parameter.num_crop)]
	price_crop_mape = [[] for i in range(Parameter.num_crop)]
	price_crop = [[] for i in range(Parameter.num_crop)]
	price_init_list = []


	for l in range(len(compare_address)):
		price_init_list.append(Init_Price(str(compare_address[l])))

	for l in range (len(model_address)):
		#for m in range(len(TruePrice_month)):

		price_crop_mae_res, price_crop_mape_res, price_crop_res = RegionalDataPrediction(price_init_list[l],
																history_month,start_month_predict, TruePrice_month[l],
																		 [Traindata_address[l]], UseData_address[l],
																		 model_address[l])
		label_1.append('mae_' + str(label[l]) + '_' + str(TruePrice_month[l]))
		label_2.append('mape_' + str(label[l]) + '_'+ str(TruePrice_month[l]))
		label_3.append('price_' + str(label[l]) + '_' + str(TruePrice_month[l]))

		for i in range(len(price_crop_mae)):
			for j in range(len(price_crop_mae_res[i])):

				price_crop_mae[i].append(price_crop_mae_res[i][j])
				price_crop_mape[i].append(price_crop_mape_res[i][j])
				price_crop[i].append(price_crop_res[i][j][-1])

	address = '新发地数据/'
	price_init = Init_Price(address)
	for i in range(Parameter.num_crop):
		price_crop[i].insert(0, price_init[i])
	label_3.insert(0, 'price_initial_BeiJing')

	address = '宁波数据/'
	price_init = Init_Price(address)
	for i in range(Parameter.num_crop):
		price_crop[i].insert(0, price_init[i])
	label_3.insert(0, 'price initial_NingBo')

	#画图和保存数据
	x = [v for v in range(len(price_crop_mae[0][0]))]
	for i in range(Parameter.num_crop):
		# Where to save the figures
		IMAGES_PATH = "./Test_4/images"
		os.makedirs(IMAGES_PATH, exist_ok=True)


		plt.figure()
		# plt.title('Month Price')
		plt.tight_layout()

		simulation_plot.simulation_plot_9(x, price_crop_mae[i], label_1, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mae")
		plt.legend()

		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mae")

		plt.figure()
		simulation_plot.simulation_plot_9(x, price_crop_mape[i], label_2, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mape")
		plt.legend()

		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mape")

		plt.figure()
		simulation_plot.simulation_plot_9(x, price_crop[i], label_3, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Price")
		plt.legend()

		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_price")

		make_dir('./Test_4/')
		filename1 = './Test_4/' + str(Parameter.vegetable[i]) + '_mae_test4.csv'
		filename2 = './Test_4/' + str(Parameter.vegetable[i]) + '_mape_test4.csv'
		filename3 = './Test_4/' + str(Parameter.vegetable[i]) + '_price_test4.csv'
		for j in range(len(price_crop_mae[i][0])):
			mae = []
			mape = []
			p = []
			for m in range(len(TruePrice_month )):#宁波和新发地数据
			# for m in range(len(TruePrice_month ) ):#宁波和新发地数据
				mae.append(price_crop_mae[i][m][j])
				mape.append(price_crop_mape[i][m][j])

			for m in range(len(price_crop[i])):
				p.append(price_crop[i][m][j])

			mae.insert(0,x[j])
			mape.insert(0,x[j])
			p.insert(0,x[j])
			SaveData(mae, filename1, ['x']+label_1, j)
			SaveData(mape, filename2, ['x']+label_2, j)
			SaveData(p, filename3, ['x'] + label_3, j)


def PricePredectTest_5(history_month, start_month_predict, TruePrice_month,
					   compare_address,UseData_address,Traindata_address,model_address,label):#5.混合地方数据预测
	# history_month为训练用参数，根据history_month个月数据预测下个月数据
	# start_month_predict为根据TruePrice_month月的数据训练好模型后，从start_month_predict个月开始预测模型
	# TruePrice_month为用多少个月的数据开始训练
	# compare_address为对比组原始数据；UseData_address为使用哪里的数据开始预测
	# 北京和宁波的模型训练，对同一组宁波数据进行预测；Traindata_address为训练数据
	# model_address为采用哪里的模型

	label_1 = []
	label_2 = []
	label_3 = []
	price_crop_mae = [[] for i in range(Parameter.num_crop)]
	price_crop_mape = [[] for i in range(Parameter.num_crop)]
	price_crop = [[] for i in range(Parameter.num_crop)]
	price_init_list = []

	for l in range(len(compare_address)):
		price_init_list.append(Init_Price(str(compare_address[l])))

	for l in range (len(model_address)):
		#for m in range(len(TruePrice_month)):

		price_crop_mae_res, price_crop_mape_res, price_crop_res = MixedRegionalDataPrediction(price_init_list[l], history_month,start_month_predict, TruePrice_month[l],
																		 [Traindata_address[l]], UseData_address[l],
																		 model_address[l])
		label_1.append('mae_' + str(label[l]) + '_' + str(TruePrice_month[l]))
		label_2.append('mape_' + str(label[l]) + '_'+ str(TruePrice_month[l]))
		label_3.append('price_' + str(label[l]) + '_' + str(TruePrice_month[l]))

		for i in range(len(price_crop_mae)):
			for j in range(len(price_crop_mae_res[i])):

				price_crop_mae[i].append(price_crop_mae_res[i][j])
				price_crop_mape[i].append(price_crop_mape_res[i][j])
				price_crop[i].append(price_crop_res[i][j][-1])

	address = '新发地数据/'
	price_init = Init_Price(address)
	for i in range(Parameter.num_crop):
		price_crop[i].insert(0, price_init[i])
	label_3.insert(0, 'price_initial_BeiJing')

	address = '宁波数据/'
	price_init = Init_Price(address)
	for i in range(Parameter.num_crop):
		price_crop[i].insert(0, price_init[i])
	label_3.insert(0, 'price initial_NingBo')

	#画图和保存数据
	x = [v for v in range(len(price_crop_mae[0][0]))]
	for i in range(Parameter.num_crop):
		# Where to save the figures
		IMAGES_PATH = "./Test_5/images"
		os.makedirs(IMAGES_PATH, exist_ok=True)


		plt.figure()
		# plt.title('Month Price')
		plt.tight_layout()

		simulation_plot.simulation_plot_9(x, price_crop_mae[i], label_1, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mae")
		plt.legend()

		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mae")

		plt.figure()
		simulation_plot.simulation_plot_9(x, price_crop_mape[i], label_2, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Mape")
		plt.legend()

		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_mape")

		plt.figure()
		simulation_plot.simulation_plot_9(x, price_crop[i], label_3, Parameter.vegetable[i])
		plt.xlabel("Month")
		plt.ylabel("Price")
		plt.legend()

		# 保存图片名
		Save_Fig(IMAGES_PATH, str(Parameter.vegetable[i]) + "_price")

		make_dir('./Test_5/')
		filename1 = './Test_5/' + str(Parameter.vegetable[i]) + '_mae_test5.csv'
		filename2 = './Test_5/' + str(Parameter.vegetable[i]) + '_mape_test5.csv'
		filename3 = './Test_5/' + str(Parameter.vegetable[i]) + '_price_test5.csv'
		for j in range(len(price_crop_mae[i][0])):
			mae = []
			mape = []
			p = []
			for m in range(len(TruePrice_month )):#宁波和新发地数据
				mae.append(price_crop_mae[i][m][j])
				mape.append(price_crop_mape[i][m][j])

			for m in range(len(price_crop[i])):
				p.append(price_crop[i][m][j])

			mae.insert(0,x[j])
			mape.insert(0,x[j])
			p.insert(0,x[j])
			SaveData(mae, filename1, ['x']+label_1, j)
			SaveData(mape, filename2, ['x']+label_2, j)
			SaveData(p, filename3, ['x'] + label_3, j)
if __name__ == "__main__":
	random.seed(Parameter.Random_Seed)
	#-----------
	# Test = 1
	# address = '宁波数据/'
	# TrainPricePredict_GBDT(Parameter.History_Month,address)

	# -----------
	# Test = 1
	# address = ['宁波数据/','新发地数据/']
	# TrainPricePredict_GBDT_AccodingData(Parameter.History_Month, 20, address)
	#
	# # -----------
	# Test = 1
	# data_address = ['宁波数据/','宁波数据/']
	# model_address = ['宁波数据/','新发地数据/']
	# label_reg = ['NingBo', 'BeiJing']
	#
	# start_month_predict = 20
	# PricePredectTest_0(start_month_predict, data_address,model_address,label_reg)#测试一下不停的预测结果

	# -----------
	# PricePredectTest_1([5,10,20,30],30)#1.随着数据量的增多，mape等指标变化，图和数据保存

	#-----------
	# Test = 1
	# PricePredectTest_2(Parameter.History_Month, [20,40,60,80,100])#2.对着某一个预测模型，多少个周期后会变的慢慢不准了.
	#3.	随着数据量的增多得到的预测模型用来不停的预测的时候，多少个周期会慢慢变的不准了

	# -----------
	# PricePredectTest_3(10)

	# -----------
	# Test = 1
	# # compare_address为对比组原始数据；UseData_address为使用哪里的数据开始预测
	# compare_address = UseData_address = ['宁波数据/', '宁波数据/']
	# #compare_address = UseData_address = ['新发地数据/','新发地数据/']
	# #compare_address = UseData_address = ['宁波数据/','新发地数据/']
	# # 北京和宁波的模型训练，对同一组宁波数据进行预测；Traindata_address为训练数据
	# # TruePrice_month, TrainData_address,model_address对应上数据
	# Traindata_address = ['宁波数据/', '新发地数据/']
	# # model_address为采用哪里的模型
	# model_address = ['宁波数据/', '新发地数据/']
	# TruePrice_month = [20, 20]
	#
	# start_month_predict = 20# start_month_predict为根据TruePrice_month月的数据训练好模型后，从start_month_predict个月开始预测模型
	# label = ['NingBo', 'BeiJing']
	# PricePredectTest_4(Parameter.History_Month,start_month_predict,TruePrice_month ,compare_address, UseData_address,Traindata_address,  model_address,label )

	# -----------
	# Test = 1
	# # compare_address为对比组原始数据；UseData_address为使用哪里的数据开始预测
	# compare_address = UseData_address = ['宁波数据/', '宁波数据/']
	# # compare_address = UseData_address = ['新发地数据/','新发地数据/']
	# # compare_address = UseData_address = ['宁波数据/','新发地数据/']
	# # 北京和宁波的模型训练，对同一组宁波数据进行预测；Traindata_address为训练数据
	# # TruePrice_month, TrainData_address,model_address对应上数据
	# Traindata_address = [['宁波数据/'], ['新发地数据/','宁波数据/']]
	# # model_address为采用哪里的模型
	# #model_address = ['宁波数据/', '新发地数据/']
	# model_address = ['宁波数据/', '混合数据模型/']
	# # TruePrice_month = [[30], [60, 30]]
	# # start_month_predict = 30
	# TruePrice_month = [[80], [120, 80]]
	# start_month_predict = 80  # start_month_predict为根据TruePrice_month月的数据训练好模型后，从start_month_predict个月开始预测模型
	# label = ['NingBo', 'BeiJing']
	# PricePredectTest_5(Parameter.History_Month, start_month_predict, TruePrice_month, compare_address, UseData_address,
	# 				   Traindata_address, model_address, label)
	#
	# Draw_Init_Price(['宁波数据/', '新发地数据/'], ['price_initial_Ningbo','price_initial_BeiJing'])


	# x_axis_data = [ 2, 3, 4, 5, 6, 7]  # x
	# y_axis_data = [ 69, 79, 71, 80, 70, 66]  # y
	#
	# plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
	# ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
	#
	# plt.legend()  # 显示上面的label
	# plt.xlabel('time')  # x_label
	# plt.ylabel('number')  # y_label
	# plt.xlim(0, 8)  # 仅设置y轴坐标范围




	plt.show()




