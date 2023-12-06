import csv
from matplotlib import pyplot as plt
import pandas as pd
from Parameter import vegetable, vagetable_chinese_name

def month_avercal(filename):
	print('month average calculation for',filename[5:-4])
	data = []
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		#print(type(reader))
		
		for row in reader:
			data.append([row[0],float(row[2])])

	data_month = []
	sum_month = 0
	current_month = 1
	month_lenth = 0
	for i in range(len(data)):
		date_month = data[i][0].split('/')[1]
		if int(date_month) != current_month :
			data_month.append(sum_month/month_lenth)
			sum_month = 0
			month_lenth = 0
			current_month = int(date_month)
		else:
			sum_month += data[i][1]
			month_lenth += 1
	#print(sum_month,month_lenth)
	data_month.append(sum_month/month_lenth)
	#print(len(data_month))
	return data_month

def std_cal(m):
	mid = sum(m)/len(m)
	s = 0
	for i in m:
		s += (i-mid)**2
	return ((s/len(m)) ** 0.5)/mid
	#return (s/len(m)) ** 0.5


def Calculate_Actual_Month_Price(num_month):
	for i in range(len(vegetable)):
		filename = '完整数据-'+ vegetable[i] +'.csv'
		data = month_avercal(filename)
		#print(data)
		filename_w = vagetable_chinese_name[i] + '的月均价格-' + vegetable[i] + '.csv'

		for j in range(len(data)):
			year = 2014+j//12
			month = j%12+1
			date_of_data = str(year) + '/' + str(month) + '/1'
			result = pd.DataFrame([[date_of_data,vegetable[i],data[j]]], columns=['date', 'name', 'price'])
			if j == 0:
				result.to_csv(filename_w, mode='w', index=False, encoding='gbk')  # header参数为None表示不显示表头
			else:
				result.to_csv(filename_w, mode='a', index=False, encoding='gbk', header=None)  # header参数为None表示不显示表头
		print(vegetable[i], 'is done!')









