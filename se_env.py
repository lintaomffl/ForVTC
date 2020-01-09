from utils import transfer_dict, cells_list, dis_hop
import numpy as np
import math

# Basestation的相关资源未定
OutLineW = 100 # 设定边缘服务器出口带宽为100Mbps
ClockRate = 2000000 # 时钟周期为 2GHz
WirelessW = 500 # 500MB
CostMrg = 0.1 # 迁移代价
Delay_Threshold = 0.15 # 时延的一个门限值

class SE_Env:
    def __init__(self,maze,qmaze,datasize_base):
        self.maze = maze
        self.datasize_base = datasize_base
        self.n_vehicles = qmaze.n_vehicles
        # SE的起始放置位置，初始位置默认为与汽车所在起始位置一致
        self.SE_ogpos_list = qmaze.vehs_og_list
        self.SE_curpos_list = qmaze.vehs_og_list
        # self.SE_data_list = np.zeros((self.n_vehicles))
        # 统计各个各自的SE的数目
        self.SE_cellnum_list = np.zeros((19))
        # 统计各个SE与车之间的时延，后续用来与Delay_Threshold比较，来计算时延情况的reward
        self.SE_delay_list = np.zeros((self.n_vehicles))
        # 统计各个SE所累积的reward
        self.game_acc_se_reward = np.zeros((self.n_vehicles))
        self.SE_total_reward_list = np.zeros((self.n_vehicles))
        # Basestation的相关资源设定
        self.BS_outlineW = OutLineW
        self.BS_wirelessW = WirelessW
        self.BS_computeR  = 1
        # self.Veh_cellnum_list = qmaze.count_cells_vehsnum()

        # 记录服务时延情况:
        self.record_delay_list = [[] for veh in range(self.n_vehicles)]
        self.record_success_rate = [[] for veh in range(self.n_vehicles)]

        self.se_step_reward_list= [0] * self.n_vehicles
        # 记录下来SE会改变的位置
        self.SEs_next_mrg_list = [-1]*self.n_vehicles

        # 生成车与SE所需要交互的数据
        self.data_size()
        # 生成各个车辆对应的SE的计算任务的CPU周期数目
        self.CPU_need()

    def reset(self,qmaze):
        self.SE_curpos_list = qmaze.vehs_og_list
        # self.SE_data_list = np.zeros((self.n_vehicles))
        # 相关信息重置\
        self.game_acc_se_reward = np.zeros((self.n_vehicles))
        self.SE_total_reward_list = np.zeros((self.n_vehicles))
        self.SE_cellnum_list = np.zeros((19))
        self.SE_delay_list = np.zeros((self.n_vehicles))
        # 记录下来SE会改变的位置
        self.SEs_next_mrg_list = [-1]*self.n_vehicles
        # 服务时延统计重新清空:
        self.record_delay_list = [[] for veh in range(self.n_vehicles)]
        self.record_success_rate = [[] for veh in range(self.n_vehicles)]
        # 重新生成veh与se需要交互的数据大小
        self.data_size()
        # 重新生成各个veh对应的SE的计算任务所需要的时钟周期数目
        self.CPU_need()
        pass

    def act(self,seactions,qmaze):
        # act函数主要进行计算reward计算、根据seaction更新位置信息、更新环境信息
        # reward计算，并且获得对应的一些需要的计算信息
        # 用来获知各个网格内车辆数目多少。该数据用来体现网格内基站无线传输资源的占用情况
        nVehsCell_list = qmaze.count_cell_vehsnum_list()
        nSEsCell_list = self.SE_count_list(qmaze)
        # 获取各个车辆所在的位置。该数据后续用来计算SE与车交互计算数据的一些时延计算。
        vehcurpos = qmaze.vehs_cur_list
        # 获取汽车的状态。该数据用来决定SE是否需要进行迁移的决策。
        veh_status_list = qmaze.status_list
        self.data_size()
        # se_step_reward_list = [self.SE_get_reward(veh,\
        #                                           veh_status_list[veh],\
        #                                           nVehsCell_list[transfer_dict[vehcurpos[veh]]],\
        #                                           nSEsCell_list[transfer_dict[self.SE_curpos_list[veh]]],\
        #                                           vehcurpos[veh],\
        #                                           self.SE_curpos_list[veh],\
        #                                           self.SE_data_list[veh],qmaze
        #                                           ) for veh in range(self.n_vehicles)]
        for veh in range(self.n_vehicles):
            self.SE_total_reward_list[veh] += self.se_step_reward_list[veh]
        # 更新SE的位置信息
        self.update_se_pos(seactions)
        seenvstates_list = [self.SE_observe(veh,qmaze) for veh in range(self.n_vehicles)]

        return seenvstates_list

    def CPU_need(self):
        # 设定各车SE所占用的CPU的时钟周期数在[4000,8000)之间
        self.CPUNeedList = 1000 * np.random.randint(4, 8, size=(self.n_vehicles))

    def update_se_pos_(self,veh):
        # 根绝SEs_next_to_mrg列表的第veh辆汽车的动作更新SE位置即可
        # 识别GOING的迁移动作,遇到GOING就不迁移，主要是针对lazy的完善
        if self.SEs_next_mrg_list[veh] != -2 :
            # if self.SEs_next_mrg_list[veh] and veh == 1:
            #     print("==更新了吗==")
            newrow,newcol = cells_list[int(self.SEs_next_mrg_list[veh])]
            self.SE_curpos_list[veh] = (newrow,newcol)

    def update_se_pos(self,seactions):
        # 根据seaction信息更新SE的位置
        for veh in range(self.n_vehicles):
            if seactions[veh] != -1: # 当action为-1，代表没有必要进行迁移
                newrow,newcol = cells_list[int(seactions[veh])]
                self.SE_curpos_list[veh] = (newrow,newcol)

    def SE_observe(self, i,qmaze):
        canvas = self.SE_draw_env(i,qmaze)
        # 将state信息转换成对应的形式，输入网络的卷积层
        se_envstate = canvas[np.newaxis, :]
        return se_envstate

    def SE_draw_env(self, i, qmaze):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape

        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0

        # draw the current position
        row, col = self.SE_curpos_list[i]
        canvas[row, col] += 1#将当前SE所在位置+1
        vrow, vcol = qmaze.vehs_cur_list[i]
        canvas[vrow, vcol] += 2#将当前对应veh所在位置+2
        return canvas

    def SE_count(self,qmaze):
        # 更新各个网格的SE数目，形成通道性的输入
        cells_SEs_num = np. zeros((5,5))
        #统计各个网格内汽车数目
        #屏蔽掉无用的网格
        #TODO 这里把无用网格屏蔽为0 ，有待尝试置-1
        for r in range(5):
            for c in range(5):
                if (r,c) not in cells_list :
                    cells_SEs_num[(r,c)] = 0
        # todo SE当前位置存储形式
        for veh in range(self.n_vehicles):
            #遍历每辆车的当前位置，并对其进行记录
            if veh in qmaze.done_list:
                continue
            r,c =self.SE_curpos_list[veh]
            cells_SEs_num[r,c] +=1
        cells_SEs_num = cells_SEs_num[np.newaxis,:]
        #print('cellsshape:',cells_vehs_num.shape())
        return cells_SEs_num

    def SE_get_reward(self,veh,veh_status,nVehsCell,nSEsCell,veh_pos,se_pos, bv, qmaze):
        # 先计算迁移以前几个单位时间delay代价
        # 再加上迁移代价
        # 在根据以上两者计算出reward
        if veh_status == -2:#说明还在行进,所以继续累计延时即可。
            self.SE_delay_list[veh] += self.sum_delay(nVehsCell, nSEsCell, veh_pos, se_pos, bv)
            return 0
        elif veh_status == -1:  # 说明失败了，可以不计算reward
            return 0
        else:# veh_status == 1 or 其他动作迁移数，说明结束到达终点了，需要计算reward
            acc_delay = self.SE_delay_list[veh] + self.sum_delay(nVehsCell, nSEsCell, veh_pos, se_pos, bv)
            avg_delay = acc_delay / qmaze.action_times_list[veh]
            if avg_delay > Delay_Threshold :
                delay_cost = - 0.20
            else:
                delay_cost = - 0.05
            self.SE_delay_list[veh] = 0  # 若计算了delaycost 要重新清零delay累加
            return delay_cost + -1 * CostMrg

    def SE_count_list(self, qmaze):
        # 维护se数量的表格，更新se_list的数量
        SEs_num_list = np.zeros(19)
        for veh in range(self.n_vehicles):
            if veh in qmaze.done_list:
                continue
            cur_idx = transfer_dict[self.SE_curpos_list[veh]]
            SEs_num_list[cur_idx] += 1
        return SEs_num_list


    def data_size(self):
        # 用来更新每一个小车每个单位时间所需要计算的数据包的大小
        #self.SE_data_list = np.random.randint(5,size=self.n_vehicles)
        # 设定每次生成的数据量大小为[0.6,1)Mb
        # self.SE_data_list = 0.1 * np.random.randint(6, 11, size=(self.n_vehicles))
        self.SE_data_list = self.datasize_base * np.ones((self.n_vehicles))


    def sum_delay(self,nVehsCell, nSEsCell, veh_pos, se_pos, bv, cpuneed):
        return self.Delay_C(nSEsCell, cpuneed)+ self.Delay_N(veh_pos,se_pos,bv)+self.Delay_WT_(nVehsCell,bv)

    def Delay_WT(self,nVehsofCell,bv_t):
        # 计算数据包的无限传输时延
        # 对应基站所含有的车辆nVehsofCell,无线传输的总带宽，每个单位产生的需要被计算的数据量
        return 1*bv_t/(WirelessW/nVehsofCell)

    def Delay_WT_(self, nVehsofCell, bv_t):
        # 计算数据包的无限传输时延
        # 对应基站所含有的车辆nVehsofCell,无线传输的总带宽，每个单位产生的需要被计算的数据量
        # return 1 * bv_t / (WirelessW / nVehsofCell)
        # 计算汽车与宏基站距离

        SNRv = (0.2 * 0.6 * 500 ** (3.25 * -0.75)) / 3.9810717055349565e-21
        if nVehsofCell == 0:
            nVehsofCell = 1
        Rv = 180000 * ( 50 / nVehsofCell) * math.log2(1 + SNRv)
        # if Rv == 0:
        #     print("***********Error:Rv=0*************\nnVehCell:",nVehsofCell)
        # print(SNRv, Rv, nVehsofCell)
        # print(bv_t*1000000/Rv)
        return bv_t*1000000/Rv

    def Delay_C(self,nSEsCell,CPUneed):
        # 计算服务计算时延
        # 对应基站所含有的车辆nVehsofCell
        return nSEsCell*CPUneed*(1/ClockRate)

    def Delay_N(self,veh_pos,se_pos,bv_t):
        # 数据包在网络线路中传输的时延
        # 需要veh_pos se_pos(0-18的形式),bv_t是每个单位时间产生的需要被计算的数据量,outlineW基站向外传输数据的光纤带宽
        return (bv_t/OutLineW) + 2*0.02*dis_hop(veh_pos,se_pos)

    def Cost_Mrg(self):
        # SE的迁移所需要服务器付出的代价
        return 1*CostMrg


    def get_se_cost(self, se_cost_list, qmaze):
        # 用来记录会得到cost的格子的队列
        cost_area_list = []
        # 统计一下各个格子数目中的cost数目
        SE_list = self.SE_count_list(qmaze)
        for i in range(19):
            if SE_list[i] >= 10 :
                cost_area_list.append(i)

        for veh in range(self.n_vehicles):
            if transfer_dict[self.SE_curpos_list[veh]] in cost_area_list:
                se_cost_list[veh] += -0.15
                # se_cost_list[veh] += -0.005 * SE_list[transfer_dict[self.SE_curpos_list[veh]]] * 1
