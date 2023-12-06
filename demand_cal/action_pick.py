import random
import torch
from common.model import MLP
# from environment.env_v4 import MyEnv
from environment.farmer import farmer
import os

class ActionPick:
    def __init__(self):
        random.seed(1)


    def random_pick(self):
        return random.randint(1, 13)

    def profit_pick(self, price):
        plant_cost = [1, 1.72, 1.63, 1.29, 1.5, 0.67, 0.42, 1.25, 1.07, 2.94, 1.94, 1.71, 0.86]
        p = []
        for i in range(13):
            p.append(price[i] - plant_cost[i])
        return p.index(max(p))+1

    def rl_pick(self, state, n_states, n_actions,model_path):
        self.target_net = MLP(n_states,n_actions, hidden_dim = 256)
        # pthfile = r'D:\Python_program\DQN_crop_rotation_v9\outputs\crop planning\20220509-235616\models\dqn_checkpoint.pth'
        #WHY
        # curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
        # parent_path = os.path.dirname(curr_path)  # 父路径
        # pthfile = parent_path + "\\outputs\\crop planning\\20220509-235616\\models\\dqn_checkpoint.pth"
        pthfile = model_path + "dqn_checkpoint.pth"
        self.target_net.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            q_values = self.target_net(state)
            action = q_values.max(1)[1].item()
        return action
    def rl_pick_static(self, state, n_states, n_actions,model_path):
        self.target_net = MLP(n_states,n_actions, hidden_dim = 256)
        # pthfile = r'D:\Python_program\DQN_crop_rotation_v9\outputs\crop planning\20220509-235616\models\dqn_checkpoint.pth'
        pthfile = r'D:\Python_program\DQN_crop_rotation_v9\outputs_dynamic\crop planning\Multi_coperative\1Tune_month_1Pre_month_1Month\models\dqn_checkpoint.pth'
        #WHY
        # curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
        # parent_path = os.path.dirname(curr_path)  # 父路径
        # pthfile = parent_path + "\\outputs\\crop planning\\20220509-235616\\models\\dqn_checkpoint.pth"
        #pthfile = model_path + "dqn_checkpoint.pth"
        self.target_net.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32)
            q_values = self.target_net(state)
            action = q_values.max(1)[1].item()
        return action



if __name__ == "__main__":
    m = ActionPick()
    farmer_all = []
    for i in range(50):
        farmer_all.append(farmer(i, random.random() + 1))
    # env = MyEnv(farmer_all)
    # obs = env.reset()
    print(m.rl_pick(obs))