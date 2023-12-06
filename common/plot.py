import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
def chinese_font():
    #return FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc',size=15)  # 系统字体路径，此处是mac的
    return FontProperties(fname='C:/Windows/Fonts/HGOCR_CNKI.ttf', size=15)
def plot_rewards(rewards,ma_rewards,tag="train",env='CartPole-v0',algo = "DQN",save=True,path='./'):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo,env))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.plot(ma_rewards,label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(path+"{}_rewards_curve".format(tag))
    # plt.show()

def plot_supply(supply,  tag="train", env='CartPole-v0', algo = "DQN", save=True, path='./'):
    sns.set()
    plt.title("the supply change of {} for {}".format(algo, env))
    vegetable = ['potato', 'tomato', 'cucumber','pakchoi','broccoli','cabbage','turnip','lettuce','chinese_watermelon','green_bean','green_pepper','eggplant','celery']
    for i in range(len(vegetable)):
        plt.plot(supply[i+1], label=vegetable[i])
    plt.legend()
    if save:
        plt.savefig(path+"{}_supply_curve".format(tag))
    plt.ylim(0, 20)
    plt.show()


def plot_rewards_cn(rewards,ma_rewards,tag="train",env='CartPole-v0',algo = "DQN",save=True,path='./'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(env,algo),fontproperties=chinese_font())
    plt.xlabel(u'回合数',fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励',u'滑动平均奖励',),loc="best",prop=chinese_font())
    if save:
        plt.savefig(path+f"{tag}_reward_curve_cn")
    # plt.show()

def plot_losses(losses,algo = "DQN",save=True,path='./'):
    sns.set()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses,label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()