import random, copy
import numpy as np
from utils import cells_list, transfer_dict
from env import *

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1

class Qmaze(object):
    """
    对于外界而言，Qmaze 是一个整体，屏蔽了类中各个 actor 的运行细节，外界不需要也不应该知道内部如何运转
    多机学习任务:
    要求:
    1. 添加距离矩阵，从区域 a 到区域 b 的时间不同
    2. actor 在行动的时候需要花费时间，入 a 至 b 耗时4分钟，在这期间不作响应
    """

    def __init__(self, maze, n_vehicles, time_matrix):
        """
        Qmaze 初始化:
            1. 转存 maze 矩阵作为系统环境，并记录 actor 数量
            2. 针对每个 actor 随机生成起始点和终点，起始点和终点在不同训练轮次中一致
        """
        self.maze = np.array(maze)
        self.n_vehicles = n_vehicles
        self.time_matrix = np.array(time_matrix)

        nrows, ncols = self.maze.shape

        # 生成n_vehicles辆车，以及他们的起始OD位置
        self.vehs_og_list = []
        self.vehs_dest_list = []
        # for veh in range(7):
        #     og, dest = 1, 4
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(7,14):
        #     og, dest = 4, 1
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(14,21):
        #     og, dest = 2, 5
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(21,28):
        #     og, dest = 5, 2
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(28,34):
        #     og, dest = 3, 6
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(34,40):
        #     og, dest = 6, 3
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])

        # for veh in range(5):
        #     og, dest = 1, 4
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(5,10):
        #     og, dest = 4, 1
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(10,15):
        #     og, dest = 2, 5
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(15,20):
        #     og, dest = 5, 2
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(20,25):
        #     og, dest = 3, 6
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(25,30):
        #     og, dest = 6, 3
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        for veh in range(9):
            og, dest = 1, 4
            self.vehs_og_list.append(cells_list[og])
            self.vehs_dest_list.append(cells_list[dest])
        for veh in range(9,18):
            og, dest = 4, 1
            self.vehs_og_list.append(cells_list[og])
            self.vehs_dest_list.append(cells_list[dest])
        for veh in range(18,26):
            og, dest = 2, 5
            self.vehs_og_list.append(cells_list[og])
            self.vehs_dest_list.append(cells_list[dest])
        for veh in range(26,34):
            og, dest = 5, 2
            self.vehs_og_list.append(cells_list[og])
            self.vehs_dest_list.append(cells_list[dest])
        for veh in range(34,42):
            og, dest = 3, 6
            self.vehs_og_list.append(cells_list[og])
            self.vehs_dest_list.append(cells_list[dest])
        for veh in range(42,50):
            og, dest = 6, 3
            self.vehs_og_list.append(cells_list[og])
            self.vehs_dest_list.append(cells_list[dest])

        # 60vehicles
        # for veh in range(10):
        #     og, dest = 1, 4
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(10,20):
        #     og, dest = 4, 1
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(20,30):
        #     og, dest = 2, 5
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(30,40):
        #     og, dest = 5, 2
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(40,50):
        #     og, dest = 3, 6
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])
        # for veh in range(50,60):
        #     og, dest = 6, 3
        #     self.vehs_og_list.append(cells_list[og])
        #     self.vehs_dest_list.append(cells_list[dest])

        # print(len(self.vehs_og_list))
        # print(self.vehs_dest_list)
        # print(len(self.__vehs_og_list))
        # 初始化各个汽车的积累reward
        self.total_reward_list = [0] * n_vehicles

        # status_list 用来标识各个 actor 的训练状态，以该状态来决定是否给 actor 派遣动作
        # 0     =>  表示尚未结束
        # 1     =>  表示顺利抵达终点
        # -1    =>  表示积累reward过大导致失败
        # -2     =>  表示该 actor 正在迁移，不应当派遣动作 => actor 不应该知道该状态何时结束
        self.status_list = [0] * n_vehicles

        # time_remain_list 记录各个 actor 完成迁移所需剩余时间
        self.time_remain_list = [0] * n_vehicles

        # 记录各个 actor 完成迁移经历的迁移次数
        self.action_times_list = [0] * n_vehicles

        # states_list 用于记录各个 actor 的状态信息，包含当前坐标及其start、end等状态
        self.states_list = [()] * n_vehicles

        # 各个 actor 当前所处在的位置坐标
        self.vehs_cur_list = copy.deepcopy(self.vehs_og_list)
        # FIXME: self.vehs_og_list 什么时候被清空的？
        self.__vehs_og_list = copy.deepcopy(self.vehs_og_list)
        # 记录vehs的下一个动作迁移点，用作滞后更新
        self.vehs_next_go_list = [()] *n_vehicles
        # 记录下会使得vehs改变格子的迁移动作，用作滞后动作记录
        self.vehs_change_act = [-1]*self.n_vehicles
        # 记录下会使得vehs改变格子的迁移动作，用作滞后reward记录
        self.reward_will_get = [0]*self.n_vehicles

        # 表示车辆的状态：
        # 0 ===> 刚开始/进行中
        # 1 ===> 到达终点
        # -1 ===> 超过最大上限，输了
        self.status_flag = [0]*self.n_vehicles
        # 表示上一个动作是否完成：
        # 0 ===> 未完成
        # 1 ===> 已经完成，初始化以后应该为1
        self.last_act_done = [1]*self.n_vehicles

        # 用来记录这一个过程的经验数据是否有必要被记录
        # 0 ===> 不需要
        # 1 ===> 需要被记录
        self.should_save = [0] * self.n_vehicles


        # 记录各个网格车辆数目
        # self.cells_vehs_num = np.zeros((5,5))
        # 记录每一轮进行训练循环里，win 以及 loss的汽车号，为了统计各个网格内的汽车数量
        self.done_list = set()
        self.arrived_list = set()

        # 记录历史动作
        self.veh_record_actions = [[]]*6


        # 用来记录每一局游戏（训练过程中的一轮）的累计的reward
        self.game_acc_reward = np.zeros((self.n_vehicles))
        self.game_acc_veh_reward = np.zeros((self.n_vehicles))
        self.game_acc_se_reward = np.zeros((self.n_vehicles))
        self.game_acc_pure_reward = np.zeros((self.n_vehicles))
        # 用来记录每辆车，每个时间片里的行车代价
        self.acc_drive_cost = np.zeros((self.n_vehicles))
        self.acc_drive_speed = np.zeros((self.n_vehicles))
        self.game_drive_cost = np.zeros((self.n_vehicles))
        self.game_drive_speed = np.zeros((self.n_vehicles))

        self.step_reward_list = np.zeros((self.n_vehicles))

        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self.maze[r, c] == 1.0]
        # self.free_cells.remove(self.target)
        if self.vehs_og_list == None:
            raise Exception("Initiate Failed : the original position didn't be initiated!")
        self.reset()

    def reset(self):
        """
        重置各个车辆的当前位置、车辆的状态、车辆的已访问列表、积累的 reward 的list
        """
        self.vehs_og_list = copy.deepcopy(self.__vehs_og_list)
        self.vehs_cur_list = copy.deepcopy(self.__vehs_og_list)
        self.vehs_next_go_list = [()]*self.n_vehicles
        self.total_reward_list = [0] * self.n_vehicles
        self.status_list = [0] * self.n_vehicles
        self.time_remain_list = [0] * self.n_vehicles
        self.action_times_list = [0] * self.n_vehicles
        self.reward_will_get = [0] *self.n_vehicles
        self.should_change = [0] * self.n_vehicles
        self.should_save = [0] * self.n_vehicles
        self.states_list = [()] * self.n_vehicles
        self.done_list.clear()
        self.arrived_list.clear()
        # 用来记录每一局游戏（训练过程中的一轮）的累计的reward,重新清零
        self.game_acc_reward = np.zeros((self.n_vehicles))
        self.game_acc_pure_reward = np.zeros((self.n_vehicles))
        self.game_acc_veh_reward = np.zeros((self.n_vehicles))
        self.game_acc_se_reward = np.zeros((self.n_vehicles))
        self.step_reward_list = np.zeros((self.n_vehicles))
        # 用来记录每辆车，每个时间片里的行车代价
        self.acc_drive_cost = np.zeros((self.n_vehicles))
        self.acc_drive_speed = np.zeros((self.n_vehicles))
        self.game_drive_cost = np.zeros((self.n_vehicles))
        self.game_drive_speed = np.zeros((self.n_vehicles))
        for veh in range(self.n_vehicles):
            row, col = self.vehs_og_list[veh]
            # self.states_list.append((row, col , 'start'))
            self.states_list[veh] = (row, col, 'start')
        self.min_reward = -0.2 * self.maze.size
        self.visited_list = [set() for veh in range(self.n_vehicles)]

        # 汽车完成状态的重置
        self.status_flag = [0] * self.n_vehicles
        # 重置上一动作完成状态
        self.last_act_done = [1] * self.n_vehicles




    def update_state_single__(self, i):
        # 读取对应的next_to_go的信息
        # 把next_to_go的信息更新到当前位置
        nextrow, nextcol = self.vehs_next_go_list[i]
        self.vehs_cur_list[i] = nextrow, nextcol
        self.states_list[i] = (nextrow, nextcol, 'going')

        # 判断是否抵达终点的信息
        if self.vehs_cur_list[i] == self.vehs_dest_list[i]:
            self.done_list.add(i)
            # print("someone arrived")
            self.status_flag[i] = 1


    def get_some_will(self, action,i):
        # 主要是为了得到next_to_go & time_remain & reward_will

        # 获得next_to_go
        # 先进行动作有效性判断:
        # 获取车辆当前坐标并调用有效性函数判断
        nrow, ncol, nmode = self.states_list[i]
        valid_actions = self.valid_actions((nrow, ncol))

        orow, ocol = nrow, ncol

        if action in valid_actions:
            # 如果动作有效:
            nmode = 'valid'
            # nrow, ncol = self.vaild_move(nrow,ncol,action)
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
            if action == UPL:
                nrow -= 1
                ncol -= 1
            elif action == DOWNR:
                nrow += 1
                ncol += 1
            # 以上处理以后，ncol与nrow即为next_to_go
            self.vehs_next_go_list[i] = (nrow, ncol)
            # 得到next_to_go的同时，计算到达从当前位置到next_to_go的所需时间:
            self.time_remain_list[i] = self.time_matrix[
                transfer_dict[(orow, ocol)],
                transfer_dict[(nrow, ncol)]
            ]
            # 得到reward_will_get
            self.reward_will_get[i] = self.get_will_reward(nrow, ncol, nmode, i, action)
        else:
            # 动作无效，得保持原地
            self.time_remain_list[i] = 1
            self.vehs_next_go_list[i] = (nrow, ncol)
            nmode = 'blocked'
            self.reward_will_get[i] = self.get_will_reward(nrow, ncol, nmode, i, action)

    def get_will_reward(self, cr, cl, nmode, veh, action):
        """
        反馈采取动作后的 reward:
        1. 走向已经访问过的轨迹的话 reward为-0.25；
        2. 采取的动作走向无效区域的话，反馈-0.75；
        3. 采取的动作有效，则为-0.04
        """
        tmp_reward = 0
        cur_row, cur_col = cr,cl
        drow, dcol = self.vehs_dest_list[veh]

        if (cur_row, cur_col) in self.visited_list[veh]:
            return -0.5

        if nmode == 'valid':
            # 根绝当前坐标与采取的动作推算前一时刻坐标；
            # 再参考坐标之间的转移的交通流，计算出veh移动的代价。
            pre_row, pre_col = self.backward_vaild_move(cur_row, cur_col, action)
            transport_cost = self.time_matrix[transfer_dict[(pre_row, pre_col)],
                                              transfer_dict[(cur_row, cur_col)]]
            tmp_reward = -0.1 * transport_cost  # 根据网格跨越时间代价矩阵为依据得reward
        elif nmode == 'blocked':
            #若是动作为无效动作，则停留原地，返回一个惩罚性质的reward
            return -0.5

        if cur_row == drow and cur_col == dcol:
            return 1.5 + tmp_reward

        return tmp_reward

    def backward_vaild_move(self, cur_row, cur_col, action):
        # 从当前坐标与刚执行过得动作，推出前一时刻坐标
        if action == LEFT:
            cur_col += 1
        elif action == UP:
            cur_row += 1
        if action == RIGHT:
            cur_col -= 1
        elif action == DOWN:
            cur_row -= 1
        if action == UPL:
            cur_row += 1
            cur_col += 1
        elif action == DOWNR:
            cur_row -= 1
            cur_col -= 1
        return cur_row, cur_col

    def get_action_times(self, veh):
        return self.action_times_list[veh]


    def count_cells_vehsnum(self):
        cells_vehs_num = np.zeros((5, 5))
        # 统计各个网格内汽车数目
        # 屏蔽掉无用的网格
        # TODO 这里把无用网格屏蔽为0 ，有待尝试置-1
        for r in range(5):
            for c in range(5):
                if (r, c) not in cells_list:
                    cells_vehs_num[(r, c)] = 0

        for veh in range(self.n_vehicles):
            # 遍历每辆车的当前位置，并对其进行记录
            if veh in self.done_list:
                continue
            r, c = self.vehs_cur_list[veh]
            cells_vehs_num[r, c] += 1
        cells_vehs_num = cells_vehs_num[np.newaxis, :]
        # print('cellsshape:',cells_vehs_num.shape())
        return cells_vehs_num

    def count_cell_vehsnum_list(self):
        cells_vehs_num_list = np.zeros(19)
        for veh in range(self.n_vehicles):
            if veh in self.done_list:
                continue
            cur_idx = transfer_dict[self.vehs_cur_list[veh]]
            cells_vehs_num_list[cur_idx] += 1
        return cells_vehs_num_list

    def observe(self, i):
        canvas = self.draw_env(i)

        # 将state信息转换成对应的形式，输入网络的卷积层
        envstate = canvas[np.newaxis, :]
        return envstate

    def draw_env(self, i):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape

        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0

        # draw the rat
        row, col, _ = self.states_list[i]
        canvas[row, col] = cur_mark

        valid_actions = self.valid_actions((row, col))
        for action in valid_actions:
            if action == LEFT:
                canvas[row, col - 1] += 0.5
            elif action == UP:
                canvas[row - 1, col] += 0.5
            if action == RIGHT:
                canvas[row, col + 1] += 0.5
            elif action == DOWN:
                canvas[row + 1, col] += 0.5
            if action == UPL:
                canvas[row - 1, col - 1] += 0.5
            elif action == DOWNR:
                canvas[row + 1, col + 1] += 0.5

        drow, dcol = self.vehs_dest_list[i]
        canvas[drow, dcol] = dest_mark
        return canvas

    def game_status(self, i):
        """
        返回环境信息，低于最小 reward 值或者达到终点:
        由于 actor 从 a 到 b 点，先更新了各个状态值之后，然后设置的 time_remain_list
        因此，只要 time_remain_list 未到 0 时认定为状态未完全更新，此时处于 GOING 状态；
        但是，qtrain 中在获取训练数据的时候，是根据 game_status 来判定是否 get_dest，如果 win
        的优先级比 going 低的话，在这里将会得到一个 going 状态，即判定 get_dest 为 False，这显然
        是错误数据，会影响模型的收敛。
        return:
        -1  => lose(失败结束)
        -2  => going(正在迁移)
        1   => win(到达终点)
        0   => not_over(未结束)
        """
        if self.total_reward_list[i] < self.min_reward:
            return -1
        if self.time_remain_list[i] > 0:
            return -2
        cur_row, cur_col, _ = self.states_list[i]
        drow, dcol = self.vehs_dest_list[i]
        if cur_row == drow and cur_col == dcol:
            # 这就是当 actor 的迁移时间为0，且位置在终点，则win
            self.arrived_list.add(i)
            return 1
        return 0

    def valid_actions(self, cell):
        """
        对于行动合法性进行判定，作为 reward 的计算依据
        """
        row, col = cell
        actions = [0, 1, 2, 3, 4, 5]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
            actions.remove(2)
            if col == 0:
                actions.remove(0)
            elif col == 2:
                actions.remove(3)
        elif row == nrows - 1:
            actions.remove(5)
            actions.remove(4)
            if col == 2:
                actions.remove(0)
            elif col == ncols - 1:
                actions.remove(3)
        elif col == 0:
            actions.remove(0)
            actions.remove(1)
            if row == 2:
                actions.remove(5)
        elif col == ncols - 1:
            actions.remove(3)
            actions.remove(4)
            if row == 2:
                actions.remove(2)
        if row == 1 and col == 3:
            actions.remove(2)
            actions.remove(3)
        if row == 3 and col == 1:
            actions.remove(0)
            actions.remove(5)

        return actions

    def get_veh_cost(self,veh_cost_list):
        cost_area_list = []
        # 先统计各网格的车辆数目
        veh_list = self.count_cell_vehsnum_list()
        for i in range(19):
            # 网格内的车辆数目大于5的话，将其记录。
            if veh_list[i] >= 10:
                cost_area_list.append(i)
        # 进行惩罚
        for veh in range(self.n_vehicles):
            if transfer_dict[self.vehs_cur_list[veh]] in cost_area_list:
                # print(veh_list)
                # print(transfer_dict[self.vehs_cur_list[veh]] )
                veh_cost_list[veh] += -0.15
                # veh_cost_list[veh] += -0.005 * veh_list[transfer_dict[self.vehs_cur_list[veh]]] * 1
                # print(veh_cost_list)

    def get_drive_cost(self,background_vehs, nVehList):
    # 统计衡量车行时间的指标
        for veh in range(self.n_vehicles):
            W_ = background_vehs[self.vehs_cur_list[veh]]
            vehnum = nVehList[transfer_dict[self.vehs_cur_list[veh]]]
            cost = 0.018 + 0.018*5.25*0.000001*( W_*100 + vehnum )**2
            # cost = 0.013*W_ + 0.013*W_*vehnum*vehnum*0.013*0.013
            self.acc_drive_cost[veh] += cost
            self.acc_drive_speed[veh] += 1/(cost )
        pass
