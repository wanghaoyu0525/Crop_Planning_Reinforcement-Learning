import csv
price = [[0 for i in range(12)] for j in range(13)]
from os.path import dirname, abspath
action_name = ['nothing', 'potato', 'tomato', 'cucumber','pakchoi','broccoli','cabbage','turnip','lettuce','chinese_watermelon','green_bean','green_pepper','eggplant','celery']
for i in range(13):
    filename = dirname(abspath(__file__)).replace("environment","") + '\\月均价格-'+action_name[i+1]+'.csv'
    with open(filename,'r') as csvfile:
        reader = csv.reader(csvfile)
        #price.append([float(row[2]) for row in reader])
        j = 0
        for row in reader:
            price[i][j%12] += round(float(row[2])/12,3)
            j += 1
for i in price:
    print(i)
