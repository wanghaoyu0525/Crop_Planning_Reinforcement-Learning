import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import Parameter
from price_data import decompose_GBDT_monthPredict
color_ = ['blue', 'green', 'red','purple','orange','CornflowerBlue','GreenYellow','Gold','DarkGray']
def simulation_plot(x,y,label_):
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        assert (len(x) == len(y))
        plt.figure()
        # set the figure style
        ax = plt.gca()
        # 设置周边的坐标轴的颜色为空白，如果不设置，图片四周就会有黑色的坐标线。
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        l = 0

        plt.plot(x[0], y[0][0], color='black', label='initial profit')

        l = len(x[0])

        for i in range(len(x)-1):
            plt.plot([l, l], [0, 1000], color='black', linestyle='--')
            #random pick,rl model, %rl model
            for j in range(len(y[i+1])):
                if i == len(x) - 2:
                    plt.plot(x[i+1], y[i+1][j], color=color_[j%3], label=label_[j],linestyle='--')
                else:
                    plt.plot(x[i + 1], y[i + 1][j], color=color_[j%3],  linestyle='--')
            l += len(x[i+1])-1
        #print(l)
        plt.legend(prop={'size': 20})
        plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.show()
def simulation_plot_2(x,y,label_):
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        assert (len(x) == len(y))
        plt.figure()
        # set the figure style
        ax = plt.gca()
        # 设置周边的坐标轴的颜色为空白，如果不设置，图片四周就会有黑色的坐标线。
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        l = 0


        for i in range(len(x)):
            #random pick,rl model, %rl model
            for j in range(len(y[i])):
                if i == len(x) - 1:
                    plt.plot(x[i], y[i][j], color=color_[j%3], label=label_[j],linestyle='--')
                else:
                    plt.plot(x[i], y[i][j], color=color_[j%3],  linestyle='--')
            l += len(x[i])-1
            plt.plot([l, l], [0, 1000], color='black', linestyle='--')

        plt.legend(prop={'size': 20})
        plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.show()
def get_random_color():
    """获取一个随机的颜色"""
    r = lambda: random.uniform(0,1)
    return [r(),r(),r(),1]

def simulation_plot_3(x,y,label_,title_):
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        assert (len(x) == len(y))
        #color_ = []

        for i in range(len(label_)):
            color_.append(get_random_color())
        #plt.figure()
        l = 0
        xx = np.array(y).reshape(1, -1)
        # xx = np.array(y[k]).flatten()
        xx2 = np.max(xx)
        peak = max(xx2)
        for j in range(len(y[0])):
            if j == len(y[0]) - 1:
                #plt.plot(x[0], y[0][j], color=color_[j%len(label_)], label='initial profit')
                plt.plot(x[0], y[0][j], color='black', label='initial profit')
            else:
                plt.plot(x[0], y[0][j], color=color_[j%len(label_)])
        l = len(x[0])



        for i in range(len(x)-1):
            # plt.plot([l, l], [0, 600], color='black', linestyle='--')
            plt.plot([l, l], [0, int(1.01* peak) + 50], color='black', linestyle='--', linewidth=0.5)
            #random pick,rl model, %rl model
            for j in range(len(y[i+1])):
                if i == len(x) - 2:
                    plt.plot(x[i+1], y[i+1][j], color=color_[j%len(label_)], label=label_[j],linestyle='--')
                else:
                    plt.plot(x[i + 1], y[i + 1][j], color=color_[j%len(label_)],  linestyle='--')
            l += len(x[i+1])-1
        #print(l)
        plt.legend(prop={'size': 20})
        plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.title(title_, fontsize=Parameter.MyTitleSize)

    #plt.show()
def simulation_plot_4(x,y,label_,title_): #多y
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        #assert (len(x) == len(y(0)))
        #color_ = []

        for i in range(len(label_)):
            color_.append(get_random_color())
        # color_ =['blue','green','red']
        #fg = plt.figure()

        for k in range(len(y)) :#不同y值
            plt.figure()
            #fig = fg.add_subplot(1, 3, k+1)  # type:plt.Axes
            #plt.subplot(1,3,k+1)
            l = 0
            xx = np.array(y[k]).reshape(1,-1)
            #xx = np.array(y[k]).flatten()
            xx2 = np.max(xx)
            peak = max(xx2)

            for j in range(len(y[k][0])):#不同合作社
                if j == len(y[k][0]) - 1:
                    #plt.plot(x[0], y[0][j], color=color_[j%len(label_)], label='initial profit')
                    # plt.plot(x[0], y[k][0][j], color='black', label='initial profit')
                    plt.plot(x[0], y[k][0][j], color='black')
                else:
                    plt.plot(x[0], y[k][0][j], color=color_[j%len(label_)])
            l = len(x[0])
            #color_ =['blue','green','red']


            for i in range(len(x)-1):#后面的曲线
                #plt.plot([l, l], [0, 600], color='black', linestyle='--')
                plt.plot([l, l], [0, int(1.01*peak) + 50 ], color='black', linestyle='--', linewidth=0.5)
                #random pick,rl model, %rl model
                for j in range(len(y[k][i+1])):#不同合作社
                    if i == len(x) - 2:
                        plt.plot(x[i+1], y[k][i+1][j], color=color_[j%len(label_)], label=label_[j],linestyle='--')
                    else:
                        plt.plot(x[i + 1], y[k][i + 1][j], color=color_[j%len(label_)],  linestyle='--')
                l += len(x[i+1])-1
            #print(l)
            plt.grid(alpha=0.4,linestyle='--', color='b')
            plt.legend(loc='upper left',prop={'size': 20})
            plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.title(title_[k], fontsize=Parameter.MyTitleSize)
            # 保存图片名
            decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, title_[k])

def simulation_plot_4_virtual(x,y1,y2,label_,title_): #多y
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        for i in range(len(label_)):
            color_.append(get_random_color())
        plt.figure()
        for k in range(len(y1)) :#不同y值
            l = 0
            xx = np.array(y1[k]).reshape(1,-1)
            #xx = np.array(y[k]).flatten()
            xx2 = np.max(xx)
            peak = max(xx2)

            for j in range(len(y1[k][0])):#不同合作社
                if j == len(y1[k][0]) - 1:
                    #plt.plot(x[0], y[0][j], color=color_[j%len(label_)], label='initial profit')
                    plt.plot(x[0], y1[k][0][j], color='black', label='initial profit')
                else:
                    plt.plot(x[0], y1[k][0][j], color=color_[j%len(label_)])
            l = len(x[0])
            for i in range(len(x)-1):#后面的曲线
                #plt.plot([l, l], [0, 600], color='black', linestyle='--')
                plt.plot([l, l], [0, int(1.01*peak) + 50 ], color='black', linestyle='--', linewidth=0.5)
                #random pick,rl model, %rl model
                for j in range(len(y1[k][i+1])):#不同合作社
                    if i == len(x) - 2:
                        plt.plot(x[i+1], y1[k][i+1][j], color=color_[j%len(label_)], label=label_[j])
                    else:
                        plt.plot(x[i + 1], y1[k][i + 1][j], color=color_[j%len(label_)])
                l += len(x[i+1])-1

        for k in range(len(y2)) :#不同y值
            l = 0
            xx = np.array(y2[k]).reshape(1,-1)
            #xx = np.array(y[k]).flatten()
            xx2 = np.max(xx)
            peak = max(xx2)

            for j in range(len(y2[k][0])):#不同合作社
                if j == len(y2[k][0]) - 1:
                    #plt.plot(x[0], y[0][j], color=color_[j%len(label_)], label='initial profit')
                    plt.plot(x[0], y2[k][0][j], color='black')
                else:
                    plt.plot(x[0], y2[k][0][j], color=color_[j%len(label_)])
            l = len(x[0])
            for i in range(len(x)-1):#后面的曲线
                #plt.plot([l, l], [0, 600], color='black', linestyle='--')
                plt.plot([l, l], [0, int(1.01*peak) + 50 ], color='black', linestyle='--', linewidth=0.5)
                #random pick,rl model, %rl model
                for j in range(len(y2[k][i+1])):#不同合作社
                    if i == len(x) - 2:
                        plt.plot(x[i+1], y2[k][i+1][j], color=color_[j%len(label_)], label=label_[j+len(y1[k][0])],linestyle='--')
                    else:
                        plt.plot(x[i + 1], y2[k][i + 1][j], color=color_[j%len(label_)],  linestyle='--')
                l += len(x[i+1])-1

        plt.legend(prop={'size': 20})
        plt.grid(alpha=0.4, linestyle='--', color='b')
        plt.tick_params(labelsize=20)  # 调整坐标轴数字大小
        plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
        plt.title(title_, fontsize=Parameter.MyTitleSize)

def simulation_plot_5(x,y,label_,title_):#supply_total_plot
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        assert (len(x) == len(y))
        #color_ = []

        for crop in range(Parameter.num_crop):
            color_.append(get_random_color())

        # plt.figure()
        l = 0
        xx = np.array(y).reshape(1, -1)
        # xx = np.array(y[k]).flatten()
        xx2 = np.max(xx)
        peak = max(xx2)


        # color_ =['blue','green','red']

        for i in range(len(x)):
            # plt.plot([l, l], [0, 600], color='black', linestyle='--')

            # random pick,rl model, %rl model
            for j in range(len(y[i])):
                for crop in range(Parameter.num_crop):
                    if i == len(x) - 2:
                        plt.plot(x[i], y[i][j][crop], color=color_[crop % Parameter.num_crop], label=label_[crop], linestyle='--',linewidth=1)
                    else:
                        plt.plot(x[i], y[i][j][crop], color=color_[crop % Parameter.num_crop], linestyle='--',linewidth=1)
            if i == 0:
                l += len(x[i])
            else:
                l += len(x[i])-1
            plt.plot([l, l], [0, int(1.01 * peak)], color='black', linestyle='--', linewidth=0.5)
        # plt.grid()
        #print(l)

def simulation_plot_6(x,y):#supply_total_plot直方图，统计各个合作社的各作物的频次
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        #color_ = []

        #crop_f = [0 for i in range(Parameter.num_crop)]
        for crop in range(Parameter.num_crop):
            color_.append(get_random_color())
         #   crop_f[crop] = y.count(crop + 1)

        plt.bar(x, y)
        # plt.grid()
def simulation_plot_7(x,y,label_,title_): #多y ，多tune_month数据,和plot_4类似
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        # assert (len(x) == len(y(0)))
        # color_ = []

        for i in range(len(label_)):
            color_.append(get_random_color())
        # color_ =['blue','green','red']
        # fg = plt.figure()
        plt.figure()
        for k in range(len(y)):#不同的tune_month曲线
            # fig = fg.add_subplot(1, 3, k+1)  # type:plt.Axes
            # plt.subplot(1,3,k+1)
            l = 0
            xx = np.array(y[k]).reshape(1, -1)
            # xx = np.array(y[k]).flatten()
            xx2 = np.max(xx)
            peak = max(xx2)

            for i in range(len(x[k])):#后面的曲线
                # plt.plot([l, l], [0, 600], color='black', linestyle='--')

                if i == 0: #第一段曲线
                    for j in range(len(y[k][i])):#不同合作社
                        if k == 0:  # 最后一个y值是对比值用虚线
                            if j == len(y[k][0]) - 1:
                                plt.plot(x[k][0], y[k][0][j], color='black', label='initial profit', linestyle='--')
                        else:
                            plt.plot(x[k][0], y[k][0][j], color='black', linestyle='--')
                    l += len(x[k][i])
                else:
                    # random pick,rl model, %rl model
                    for j in range(len(y[k][i])):#不同合作社

                        if i == len(x[k]) - 1:  # 每个y值的最后一个曲线用图标
                            plt.plot(x[k][i], y[k][i][j], color=color_[(k * Parameter.num_cooperative + j) % len(label_)],
                                     label=label_[k * Parameter.num_cooperative + j])
                        else:
                            plt.plot(x[k][i], y[k][i][j], color=color_[(k * Parameter.num_cooperative + j) % len(label_)])
                    l += len(x[k][i]) - 1
                #plt.plot([l, l], [0, int(1.01 * peak) + 50], color='black', linestyle='--', linewidth=0.5)
                plt.plot([l+0.2*k, l+0.2*k], [0, int(1.01 * peak) + 50], color=color_[(k * Parameter.num_cooperative + j )% len(label_)], linestyle='--', linewidth=0.5)
            #print(l)
            plt.legend(prop={'size': 20})
            # plt.grid()
            plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            #plt.tick_params(labelsize=7)  # 调整坐标轴数字大小
            plt.title(title_, fontsize=Parameter.MyTitleSize)

def simulation_plot_8(x,y,label_,title_): #多y ，多Pre_month数据,和plot_4类似
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        # assert (len(x) == len(y(0)))
        # color_ = []

        for i in range(len(label_)):
            color_.append(get_random_color())
        # color_ =['blue','green','red']
        # fg = plt.figure()
        plt.figure()
        for k in range(len(y)):#不同的Pre_month曲线
            # fig = fg.add_subplot(1, 3, k+1)  # type:plt.Axes
            # plt.subplot(1,3,k+1)
            l = 0
            xx = np.array(y[k]).reshape(1, -1)
            # xx = np.array(y[k]).flatten()
            xx2 = np.max(xx)
            peak = max(xx2)

            for i in range(len(x[k])):#后面的曲线
                # plt.plot([l, l], [0, 600], color='black', linestyle='--')

                if i == 0: #第一段曲线
                    for j in range(len(y[k][i])):#不同合作社
                        #if j == len(y[k][0]) - 1:
                        #if i == len(x[k]) - 1:  # 每个y值的最后一个曲线用图标
                        plt.plot(x[k][0], y[k][0][j], color=color_[(k * Parameter.num_cooperative + j) % len(label_)], label='initial profit', linestyle='--')
                        # else:
                        #     plt.plot(x[k][0], y[k][0][j], color=color_[(k * Parameter.num_cooperative + j) % len(label_)], linestyle='--')
                    l += len(x[k][i])
                else:
                    # random pick,rl model, %rl model
                    for j in range(len(y[k][i])):#不同合作社
                        if i == len(x[k]) - 1:#每个y值的最后一个曲线用图标
                            plt.plot(x[k][i], y[k][i][j], color=color_[(k * Parameter.num_cooperative + j) % len(label_)], label=label_[k * Parameter.num_cooperative + j])
                        else:
                            plt.plot(x[k][i], y[k][i][j], color=color_[(k * Parameter.num_cooperative + j) % len(label_)])

                    l += len(x[k][i]) - 1
                plt.plot([l+0.2*k, l+0.2*k], [0, int(1.01 * peak) + 50], color=color_[(k * Parameter.num_cooperative + j )% len(label_)], linestyle='--', linewidth=0.5)
            #print(l)
            plt.legend(prop={'size': 20})
            # plt.grid()
            plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.ylabel("Profit(10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            #plt.tick_params(labelsize=7)  # 调整坐标轴数字大小
            plt.title(title_, fontsize=Parameter.MyTitleSize)

def simulation_plot_9(x,y,label_,title_):#price_plot,多y
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题

        for crop in range(len(y)):
            color_.append(get_random_color())


        for j in range(len(y)):
            plt.plot(x, y[j], color=color_[j], label=label_[j], linestyle='--', linewidth=1)
        plt.tick_params(labelsize=Parameter.MyFontSize)  # 调整坐标轴数字大小
        # plt.grid()
        plt.title(title_, fontsize=Parameter.MyTitleSize)


def simulation_plot_10(x,y,label_,title_):#error_plot,单y
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        for crop in range(len(y)):
            color_.append(get_random_color())


        plt.plot(x, y, color=color_[0], label=label_[0], linestyle='--', linewidth=1)
        plt.title(title_, fontsize=Parameter.MyTitleSize)


def simulation_plot_11(actual_y,virtual_y):#画虚拟profit
    sns.set()
    with sns.axes_style('ticks'):  # 使用ticks主题
        # for crop in range(len(y)):
        #     color_.append(get_random_color())
        #plt.figure()
        label_ = []
        for n in range(Parameter.num_cooperative):
            # label_.append('Actual Profit, probability is ' +str(Parameter.cooperative_pro[n]))
            label_.append('Actual Result')
        label_.append('Virtual Results')
        for n in range(Parameter.num_cooperative):
            #plt.subplot(Parameter.num_cooperative, 1, n + 1)
            plt.figure()
            #actual profit
            x_start = 0
            for i in range(len(actual_y)): #第几段
                if (i == 0):
                    x = [(v + x_start) for v in range(len(actual_y[i][n]))]
                    x_start += len(actual_y[i][n])
                    pmaker1, = plt.plot(x, actual_y[i][n], color=color_[n], label='Actual Result', linewidth=1.5)
                else:
                    x = [(v + x_start - 1) for v in range(len(actual_y[i][n]))]
                    x_start += len(actual_y[i][n]) - 1
                    plt.plot(x, actual_y[i][n], color=color_[n], linewidth=1.5)

            # virtual profit

            index = 0
            l = []
            peak = 0
            for key, value in virtual_y[n].items():#第几段
                color_.append(get_random_color())
                x = [(v + key) for v in range(len(value))]
                # pmaker2, = plt.plot(x, value, color=color_[Parameter.num_cooperative + index], label='Virtual Results',linestyle='--', linewidth=0.5)
                pmaker2, = plt.plot(x, value, color='gray', label='Virtual Results',linestyle='--', linewidth=0.5)
                index += 1
                l.append(key)
                xx = np.array(value).reshape(1, -1)
                # xx = np.array(y[k]).flatten()
                xx2 = np.max(xx)
                peak = max(xx2, peak)
                plt.plot([key, key], [0, int(1.01 * peak)], color='black', linestyle='--', linewidth=0.5)
            plt.title('Virtual and Actual Total Profit', fontsize=Parameter.MyTitleSize)
            # plt.legend(loc = 'upper left',prop={'size': 20})
            plt.legend(handles=[pmaker1,pmaker2],labels=['Actual Result','Virtual Results'],  loc = 'upper left',prop={'size': 20})
            plt.tick_params(labelsize=20)  # 调整坐标轴数字大小
            plt.grid(alpha=0.5, axis='y', linestyle='--', color='b')
            plt.xlabel("Month",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            plt.ylabel("Profit (10^3 CNY)",{'fontname':'Times New Roman','fontsize':Parameter.MyFontSize})
            # 保存图片名
            decompose_GBDT_monthPredict.Save_Fig(Parameter.IMAGES_PATH, 'virtual_and_actual_total_profit_' + str(n))

#以第一个像素为准，相同色改为透明
def transparent_back(img):
    img =img.convert('RGBA')
    L,H =img.size
    color_0 =img.getpixel((0,0))
    for h in range(H):
        for l in range(L):
            dot =(l,h)
            color_1 =img.getpixel(dot)
            if color_1 ==color_0:
                color_1 =color_1[:-1]+(0,)
                img.putpixel(dot,color_1)
    return img

if __name__ == '__main__':
    x_1 = [i for i in range(1,11)]
    x_2 = [i for i in range(10,23)]
    x_3 = [i for i in range(22,35)]
    x_4 = [i for i in range(34,47)]

    profit = [0.0, 69.73, 29.08, 23.11, 60.45, 26.52, 33.71, 22.27, 125.15, 71.56]


    profit_rl_2 = [44.86, 43.96, 49.44, 51.51, 61.73, 85.25, 91.01, 94.98, 108.99, 108.72, 110.29, 156.19]
    profit_sj_2 = [155.48, 155.56, 153.3, 147.25, 139.98, 116.52, 108.4, 103.84, 87.89, 87.39, 79.32, 30.27]
    profit_mix_2 = [30.35, 30.0, 32.21, 33.41, 37.93, 50.45, 53.88, 54.42, 57.5, 61.6, 65.3, 75.65]
    profit_sj_2.insert(0,profit[-1])
    profit_rl_2.insert(0,profit[-1])
    profit_mix_2.insert(0,profit[-1])

    profit_rl_3 = [81.93, 82.02, 86.03, 86.03, 112.34, 125.18, 135.01, 143.78, 159.91, 155.72, 171.32, 222.47]
    profit_sj_3 = [221.15, 220.28, 215.78, 215.89, 185.98, 164.68, 153.75, 145.95, 127.01, 129.71, 112.56, 51.48]
    profit_mix_3 = [49.29, 49.2, 50.63, 49.93, 58.79, 69.31, 80.43, 85.36, 102.5, 100.5, 119.96, 170.61]
    profit_sj_3.insert(0,profit_sj_2[-1])
    profit_rl_3.insert(0,profit_rl_2[-1])
    profit_mix_3.insert(0,profit_mix_2[-1])

    profit_rl_4 = [182.55, 187.42, 196.99, 196.84, 208.03, 217.86, 233.75, 235.51, 260.81, 259.11, 251.15, 301.2]
    profit_sj_4 = [291.03, 289.84, 286.45, 286.78, 268.41, 248.06, 219.7, 214.03, 173.82, 177.99, 165.95, 77.31]
    profit_mix_4 = [86.61, 87.67, 95.53, 96.58, 100.58, 106.29, 112.11, 113.21, 124.95, 126.45, 137.9, 193.45]
    profit_sj_4.insert(0,profit_sj_3[-1])
    profit_rl_4.insert(0,profit_rl_3[-1])
    profit_mix_4.insert(0,profit_mix_3[-1])

    x = [x_1, x_2,x_3,x_4]
    y = [[profit], [profit_sj_2, profit_rl_2, profit_mix_2],
         [profit_sj_3, profit_rl_3, profit_mix_3], [profit_sj_4, profit_rl_4, profit_mix_4]]
    label_ = ['random pick', 'rl model', '%d rl model']
    simulation_plot(x,y,label_)