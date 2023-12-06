import torch
from common.model import MLP
import random
from environment.farmer import farmer
from environment.env_v4 import MyEnv
import matplotlib.pyplot as plt
from os.path import dirname
import os


def rl_pick(state, n_states, n_actions,pthfile):
    target_net = MLP(n_states, n_actions, hidden_dim=256)
    # pthfile = r'C:\Users\fmh\Desktop\毕业设计\DQN_crop_rotation_v9\outputs\crop planning\20220510-005744\models\dqn_checkpoint.pth'
    target_net.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))
    with torch.no_grad():
        state = torch.tensor([state], dtype=torch.float32)
        q_values = target_net(state)
        action = q_values.max(1)[1].item()
    return action


if __name__ == "__main__":
    crop_name = ['nothing', 'potato', 'tomato', 'cucumber', 'pakchoi', 'broccoli', 'cabbage', 'turnip', 'lettuce',
                   'chinese_watermelon', 'green_bean','green_pepper', 'eggplant', 'celery']

    random.seed(1)
    num_farmer = 50

    farmer_all = []
    init_state = []
    for i in range(num_farmer):
        farmer_all.append(farmer(i, random.random() + 1))
        init_state.append(random.randint(1, 13))

    #dir_names = os.listdir(r'D:\Python_program\DQN_crop_rotation_v9\outputs\crop planning')
    # WHY
    pthfile = dirname(os.path.abspath(__file__)).replace("data_plot", "outputs\\crop planning")
    dir_names = os.listdir(pthfile)

    middle_profit = []

    for dir in dir_names:

        #pthfile = 'D:\\Python_program\\DQN_crop_rotation_v9\\outputs\\crop planning\\'+dir+'\\models\\dqn_checkpoint.pth'
        # WHY
        pthfile = dirname(os.path.abspath(__file__)).replace("data_plot", "outputs\\crop planning\\") + dir+'\\models\\dqn_checkpoint.pth'

        env = MyEnv(farmer_all, init_state)
        n_states = env.num_crop * 3 + 1 + 3
        n_actions = env.num_crop + 1
        obs = env.reset()

        profit_test = []
        rotation_problem_test = []
        plant_0_yeild_test = []
        supply = []
        supply_12 = []
        for e in range(120):
            action = []
            for f in range(num_farmer):
                # 根据RL模型学习到的策略
                state = []
                state.extend(obs)
                state.append(farmer_all[f].area*2-3)
                state.append(farmer_all[f].current_crop*2/13-1)
                state.append(farmer_all[f].plant_month*2/11-1)
                if env.is_current_done(env.farmer_all[f].current_crop, env.farmer_all[f].plant_month):
                    action_f = rl_pick(state, n_states, n_actions, pthfile)
                    action.append(action_f)
                else:
                    action.append(0)

            obs, reward, done, info = env.step(action)
            # profit_test.append(round(info['actual_profit'],2))
            profit_test.append(round(info['profit'], 2))
            rotation_problem_test.append(info['rotation_problem'])
            plant_0_yeild_test.append(info['plant_0_yeild'])
            supply.append(info['supply'])
            m = []
            for i in range(14):
                m.append(sum(info['supply_in_last_12'][i])/len(info['supply_in_last_12'][i]))
            supply_12.append(m)
        print(dir)
        print(profit_test)
        print(rotation_problem_test)
        print(plant_0_yeild_test)

        middle_profit.append(profit_test[60])

        x = [i for i in range(1, 121)]
        plt.plot(x, profit_test, label=dir)
        plt.legend()

    plt.show()
    m = max(middle_profit)
    s = middle_profit.index(m)
    print(m)
    print(dir_names[s])



