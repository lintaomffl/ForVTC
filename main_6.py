from __future__ import print_function
import os, sys, time, datetime, json, random, copy
import numpy as np
import pickle
from se_env import SE_Env,CostMrg,Delay_Threshold
from experience import Experience
from qmaze import Qmaze
from new_model import build_model_v2se4
from utils import dis_hop, transfer_dict, time_spend_matrix, static_shortPath, time_spend_matrix_D, Dijkstra
from utils import format_time,time_spend_matrix_veh,background_vehs, cells_list, origin_dest_action
from env import *
replace_target_iter = 100
replace_perform_iter = 100

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
Delay_Threshold = 0.35
CostMrgS = 0.2
def qtrain(model_eval_list, model_target_list,perform_vehs, epsilon=0.05, n_vehicles=20, **opt):
    # global epsilon
    epoch = 0
    n_epoch = opt.get('n_epoch', 1500)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    # name = opt.get('name', 'model')
    perform_vehs = perform_vehs
    start_time = datetime.datetime.now()
    n_vehicles = n_vehicles
    vr_weights = opt.get('vr_weights')
    get_dest_count = 0
    failed_count = 0

    eval_train_time_list = [0 for vehtype in range(6)]
    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    # if weights_file:
    #     print("loading weights from file: %s" % (weights_file,))
    #     model_eval.load_weights(weights_file)

    # Initialize experience replay object
    experience_list = [Experience(model_eval_list[vehtype], model_target_list[vehtype], max_memory=max_memory) for vehtype in range(6)]
    # experience = Experience(model_eval, model_target, max_memory=max_memory)

    # records : 记录各种指标
    # loss部分:
    records_of_loss_total = []
    records_of_loss_veh = []
    records_of_loss_se = []
    # reward部分:
    records_of_total_reward = []
    records_of_veh_reward = []
    records_of_se_reward = []
    # 这一项totalreward是veh与se_reward的单纯相加，不带权值。
    records_of_pure_total_reward = []
    # 需要输出的部分:
    records_of_veh_drive = []
    records_of_veh_drive_speed = []
    records_of_se_delay = []
    records_of_se_SR = []

    for epoch in range(n_epoch):
        loss_total = 0.0
        loss_veh = 0.0
        loss_se = 0.0

        qmaze.reset()
        # print("训练一轮结束，重置qmaze")
        # print(qmaze.vehs_og_list)
        seenv.reset(qmaze)#SE相关环境信息的重置
        game_over = False

        # get initial envstate
        envstates_list = [qmaze.observe(veh) for veh in range(n_vehicles)]
        se_envstates_list = [seenv.SE_observe(veh, qmaze) for veh in range(n_vehicles)]

        #reward_list 是用来计算SE部分和Veh行车部分两个reward的累加值，即“总的reward”。
        reward_list = np.zeros((n_vehicles))
        pure_reward_list = np.zeros((n_vehicles))
        veh_step_reward_list = np.zeros((n_vehicles))
        se_step_reward_list = np.zeros((n_vehicles))
        # 开始前把每一局将经历的step_cost_list全部置零
        # 这里的cost也是积累性质的，同reward一起反馈回去。
        veh_step_cost_list = np.zeros((n_vehicles))
        se_step_cost_list = np.zeros((n_vehicles))
        totalcost_list = np.zeros((n_vehicles))

        n_episodes = 0



        while not game_over:
            # 开始前都将需要记录这个tag置零
            qmaze.should_save = [0] * n_vehicles

            # 开始前把每一步经历的step_cost_list全部置零
            veh_step_cost_list = np.zeros((n_vehicles))
            se_step_cost_list = np.zeros((n_vehicles))

            prev_envstates_list = envstates_list
            prev_se_envstates_list = se_envstates_list
            # 获取当前各个网格车辆的信息,汽车行动前的网格汽车数目
            # _2nd_channel = background_vehs[np.newaxis,:]
            # background_vehs = background_vehs[np.newaxis,:]
            vehsnum_before_act = qmaze.count_cells_vehsnum()
            sesnum_before_act = seenv.SE_count(qmaze)

            actions = -1 * np.ones((n_vehicles)) # 默认 -1，表示不采取行动
            se_actions = -1 * np.ones((n_vehicles))  # 默认 -1，表示不采取行

            nVehsCell_list = qmaze.count_cell_vehsnum_list()
            nSEsCell_list  = seenv.SE_count_list(qmaze)
            # 每个时段的的开头部分
            # 在时段的开头先统计各个网格的车辆，
            # 从而得知，在这一个时段内，汽车们、SE们可能得到的拥挤开销
            # 这等于是在哥哥时刻开始之前计算拥挤开销被计算入reward之前的拥挤开销的积累
            qmaze.get_veh_cost(veh_step_cost_list)
            seenv.get_se_cost(se_step_cost_list, qmaze)
            # 计算一个评判行车指标drive_cost
            qmaze.get_drive_cost(background_vehs,nVehsCell_list)
            # 每个时段的的开头部分
            for veh in range(n_vehicles):

                # 寻找合适的模型
                vehtype = veh//5

                if qmaze.status_flag[veh] != 0:
                    # 说明汽车已经1或者-1了，不用在对其进行动作。
                    continue
                else:
                    # 进入这一分支说明该车还没结束
                    if qmaze.last_act_done[veh] == 1:
                    # 汽车是刚开始的状态或者完成了上一个动作的执行，需要委派新的动作。
                        valid_actions = qmaze.valid_actions(qmaze.vehs_cur_list[veh])
                        if np.random.rand() < epsilon:
                            qmaze.vehs_change_act[veh] = actions[veh] = random.choice(valid_actions)
                            seenv.SEs_next_mrg_list[veh] = se_actions[veh] = random.randint(0, 18)
                        else:
                            # print(veh,vehtype)
                            qmaze.vehs_change_act[veh] = actions[veh] = np.argmax(
                                experience_list[vehtype].predict_e(prev_envstates_list[veh], vehsnum_before_act,
                                                     prev_se_envstates_list[veh], sesnum_before_act))
                            seenv.SEs_next_mrg_list[veh] = se_actions[veh] = np.argmax(
                                experience_list[vehtype].predict_e_se(prev_envstates_list[veh], vehsnum_before_act,
                                                        prev_se_envstates_list[veh], sesnum_before_act))
                        # 重新指定time_remain_list
                        # 获取reward_will_get
                        # 还得记录一下vehs的nexttogo
                        qmaze.get_some_will(actions[veh], veh)

                    if qmaze.last_act_done[veh] == 0:
                    # 未完成上一个迁移动作，SEs和Vehs的状态都是GOING
                        actions[veh] = GOING
                        se_actions[veh] = GOING
                        continue

            # 每个时段的结尾部分
            nVehsCell_list = qmaze.count_cell_vehsnum_list()
            nSEsCell_list = seenv.SE_count_list(qmaze)
            vehcurpos = qmaze.vehs_cur_list
            # print("E:",n_episodes)
            # print("action",qmaze.vehs_change_act)
            # print("pos",vehcurpos)
            # print("next:",qmaze.vehs_next_go_list)
            # print("remain",qmaze.time_remain_list)
            # print("reward:",qmaze.game_acc_veh_reward)
            for veh in range(n_vehicles):
                # 查询汽车是否为“完成状态”，即到达终点或者累计reward过大；
                if qmaze.status_flag[veh] != 0:
                    # 说明该车状态为完成，即已到达终点或者累计reward过大
                    continue
                else:
                    # 说明汽车不是“完成”状态。
                    qmaze.time_remain_list[veh] -= 1
                    qmaze.action_times_list[veh] += 1
                    nVehsCell = nVehsCell_list[transfer_dict[vehcurpos[veh]]]
                    nSEsCell = nSEsCell_list[transfer_dict[seenv.SE_curpos_list[veh]]]
                    veh_pos = vehcurpos[veh]
                    se_pos = seenv.SE_curpos_list[veh]
                    bv = seenv.SE_data_list[veh]
                    cpuneed = seenv.CPUNeedList[veh]
                    if qmaze.time_remain_list[veh] != 0:
                        # 首先把last_act_done设置成0，表示上一个动作未做完
                        qmaze.last_act_done[veh] = 0
                        # 说明未发生迁移，所以得到的reward都为0，只是记录服务时延
                        veh_step_reward_list[veh] = 0
                        # 计算一下服务时延，
                        # 记录一下时延——后面看一下总的时延时间；
                        # 记录一下服务时延的成功率；
                        delay_time = seenv.sum_delay(nVehsCell, nSEsCell, veh_pos, se_pos, bv, cpuneed)
                        if delay_time < Delay_Threshold:
                            seenv.record_success_rate[veh].append(1)
                        else:
                            seenv.record_success_rate[veh].append(0)
                        seenv.record_delay_list[veh].append(delay_time)
                        seenv.SE_delay_list[veh] += delay_time
                        se_step_reward_list[veh] = 0
                        continue
                    if qmaze.time_remain_list[veh] == 0:
                        # 说明将要完成迁移
                        # 首先把last_act_done设置成1，表示上一个动作已经做完
                        qmaze.last_act_done[veh] = 1
                        # qmaze.should_change[veh] = 1
                        # veh的rewrd是直接读取will_get即可
                        # se部分计算时延、计算reward
                        veh_step_reward_list[veh] = qmaze.reward_will_get[veh] + veh_step_cost_list[veh]
                        qmaze.reward_will_get[veh] = 0
                        veh_step_cost_list[veh] = 0
                        # 计算一下服务时延，
                        # 记录一下时延——后面看一下总的时延时间；
                        # 记录一下服务时延的成功率；
                        delay_time = seenv.sum_delay(nVehsCell, nSEsCell, veh_pos, se_pos, bv, cpuneed)
                        if delay_time < Delay_Threshold:
                            seenv.record_success_rate[veh].append(1)
                        else:
                            seenv.record_success_rate[veh].append(0)
                        seenv.record_delay_list[veh].append(delay_time)
                        seenv.SE_delay_list[veh] += delay_time
                        avg_delay = seenv.SE_delay_list[veh] / qmaze.action_times_list[veh]
                        if avg_delay > Delay_Threshold:
                            delay_cost = - 0.20
                        else:
                            delay_cost = 0
                        seenv.SE_delay_list[veh] = 0  # 若计算了delaycost 要重新清零delay累加
                        se_step_reward_list[veh] = delay_cost + -1 * CostMrgS + se_step_cost_list[veh]
                        se_step_cost_list[veh] = 0 # 被计入reward过后的拥挤开销需要清零

                        # 计算累计的reward,加入了拥挤开销的reward
                        qmaze.game_acc_veh_reward[veh] += veh_step_reward_list[veh]
                        seenv.game_acc_se_reward[veh] += se_step_reward_list[veh]
                        pure_reward_list[veh] = veh_step_reward_list[veh] +  se_step_reward_list[veh]
                        reward_list[veh] = vr_weights * veh_step_reward_list[veh] + (1 - vr_weights) * se_step_reward_list[veh]
                        qmaze.game_acc_reward[veh] += reward_list[veh]
                        qmaze.game_acc_pure_reward[veh] += pure_reward_list[veh]
                        # 判断行车的accreward是否小于最低要求而游戏失败
                        if qmaze.game_acc_veh_reward[veh] < qmaze.min_reward:
                            # 行车失败
                            qmaze.status_flag[veh] = -1
                        else:

                            # 尚未达到失败的标准，继续行车:
                            # 根据车辆的will的得到的动作信息，更新veh与se状态。
                            # 判断一个是否到达终点的信息，即用来更新汽车状态，又用来在experience的判定因素。
                            qmaze.update_state_single__(veh)
                            seenv.update_se_pos_(veh)

                            # 维护一个should_save,表示有必要存储的经验过程数据
                            qmaze.should_save[veh] = 1
            envstates_list = [qmaze.observe(veh) for veh in range(n_vehicles)]
            se_envstates_list = [seenv.SE_observe(veh, qmaze) for veh in range(n_vehicles)]
            # 获取当前各个网格车辆的信息,汽车行动后的网格汽车数目
            vehsnum_after_act = qmaze.count_cells_vehsnum()
            sesnum_after_act = seenv.SE_count(qmaze)


            for veh in range(n_vehicles):
                if qmaze.should_save[veh] == 1:
                    # 记录一个episode经验
                    if qmaze.status_flag[veh] == 1:
                        get_dest = True
                    else:
                        get_dest = False
                    vehtype = veh//5
                    episode = [prev_envstates_list[veh], prev_se_envstates_list[veh],
                               qmaze.vehs_change_act[veh], seenv.SEs_next_mrg_list[veh],
                               reward_list[veh], veh_step_reward_list[veh], se_step_reward_list[veh],
                               envstates_list[veh], se_envstates_list[veh],
                               get_dest, vehsnum_before_act, vehsnum_after_act,
                               sesnum_before_act, sesnum_after_act]
                    experience_list[vehtype].remember(episode)
                else:
                    continue

            # 重置两个计数器
            get_dest_count = 0
            failed_count = 0
            #  todo 判断条件要改
            # 计算已经完成的 actor 数量，包括 1(成功)，2(失败)
            for veh in range(n_vehicles):
                if qmaze.status_flag[veh] == 1:
                    get_dest_count += 1
                elif qmaze.status_flag[veh] == -1:
                    failed_count += 1

            # 当所有的 actor 都已达到终点(不一定是最优解)或者未能达终点(失败)，该 episode 训练结束
            if get_dest_count + failed_count == n_vehicles:
                game_over = True
            else:
                game_over = False


            n_episodes += 1

            for vehtype in range(6):
                # Train neural network model
                inputs1, inputs2, inputs3, inputs4, targets1, targets2 = experience_list[vehtype].get_data(data_size=data_size)
                # print("datasize of traindata:",len(experience.memory))
                if len(experience_list[vehtype].memory) < 16:
                    continue
                else:
                    h = model_eval_list[vehtype].fit(
                        [inputs1, inputs2, inputs3, inputs4],
                        [targets1, targets2],
                        epochs=8,
                        batch_size=16,
                        verbose=0,
                    )
                    eval_train_time_list[vehtype] += 1
                    # print("train_",vehtype)
                    loss = model_eval_list[vehtype].evaluate([inputs1, inputs2, inputs3, inputs4], [targets1, targets2], verbose=0)
                    loss_total = loss[0]
                    loss_veh = loss[1]
                    loss_se = loss[2]

                if eval_train_time_list[vehtype] % replace_target_iter == 0:
                    experience_list[vehtype].carry_parameters_totarget()
                    print("The weights of Target NN_",vehtype," is changed")
        # 一系列reward信息的统计
        sum_pure_total_reward = 0
        sum_total_reward = 0
        sum_veh_reward = 0
        sum_se_reward = 0
        for veh in qmaze.done_list:
            sum_total_reward += qmaze.game_acc_reward[veh]
            sum_veh_reward += qmaze.game_acc_veh_reward[veh]
            sum_se_reward += seenv.game_acc_se_reward[veh]
            sum_pure_total_reward += qmaze.game_acc_pure_reward[veh]

        if len(qmaze.done_list) == 0:
            records_veh = 0
            records_total = 0
            records_se = 0
            records_total_pure = 0
        else:
            records_veh = sum_veh_reward / len(qmaze.done_list)
            records_total = sum_total_reward / len(qmaze.done_list)
            records_se = sum_se_reward / len(qmaze.done_list)
            records_total_pure  = sum_pure_total_reward /len(qmaze.done_list)
        records_of_total_reward.append(records_total)
        records_of_veh_reward.append(records_veh)
        records_of_se_reward.append(records_se)
        records_of_pure_total_reward.append(records_total_pure)
        # 统计汽车的行车代价
        sum_veh_drive = 0
        sum_veh_drive_speed = 0
        # print(qmaze.acc_drive_cost)
        for veh in qmaze.done_list:
            qmaze.game_drive_cost[veh] = qmaze.acc_drive_cost[veh] / qmaze.action_times_list[veh]
            qmaze.game_drive_speed[veh] = 1/qmaze.game_drive_cost[veh]
            sum_veh_drive += qmaze.acc_drive_cost[veh] / qmaze.action_times_list[veh]
            sum_veh_drive_speed += qmaze.game_drive_speed[veh]
        if  len(qmaze.done_list) == 0:
            records_drive = 0
            records_drive_speed = 0
        else:
            records_drive = sum_veh_drive / len(qmaze.done_list)
            records_drive_speed = sum_veh_drive_speed / len(qmaze.done_list)
        records_of_veh_drive.append(records_drive)
        records_of_veh_drive_speed.append(records_drive_speed)
        # 统计各个车辆的在每一局（每一个训练轮次）的平均delay与服务成功率
        se_delay_avg = [0]*n_vehicles
        se_success_rate = [0]*n_vehicles


        for veh in qmaze.done_list:
            se_delay_avg[veh] = sum(seenv.record_delay_list[veh])/ len(seenv.record_delay_list[veh])
            se_success_rate[veh]  = sum(seenv.record_success_rate[veh])/ len(seenv.record_success_rate[veh])

        # 在一个轮次中对于各个车辆的delay与服务成功率进行求平均
        AvgDelayforAll = 0
        SRforAll = 0
        SRcount = 0
        for veh in qmaze.done_list:
            AvgDelayforAll += se_delay_avg[veh]
            SRforAll += sum(seenv.record_success_rate[veh])
            SRcount += len(seenv.record_success_rate[veh])
        if len(qmaze.done_list) == 0:
            records_delay = 0
        else:
            records_delay = AvgDelayforAll / len(qmaze.done_list)
        if SRcount == 0:
            records_SR =0
        else:
            records_SR = SRforAll / SRcount
        records_of_se_delay.append(records_delay)
        records_of_se_SR.append(records_SR)

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Episodes: {:d} | GetDestCount: {:d}/{:d} |FailedCount: {:d}/{:d}| time: {}"
        print(template.format(epoch, n_epoch - 1,n_episodes ,get_dest_count, n_vehicles ,failed_count, n_vehicles, t))
        print("Arrived vehs:", qmaze.done_list)
        print("SE_delay_avg:",se_delay_avg)
        print("SE_success_rate:",se_success_rate)
        print("Veh_drive:",qmaze.game_drive_cost)
        print("Veh_drive_speed:",qmaze.game_drive_speed)
        print("【AVG_Veh_drive】:", records_drive)
        print("【AVG_Veh_drive_speed】:", records_drive_speed)
        # print("VEH_ACC_Reward:",qmaze.game_acc_veh_reward)
        # print("AVG_VEH:", records_veh)
        # print("SE_ACC_Reward:",seenv.game_acc_se_reward)
        # print("AVG_SE:",records_se)
        # print("AVG_Reward:",records_total)

        # 这里统一的记录一次(一轮epoch)
        records_of_loss_total.append(loss_total)
        records_of_loss_veh.append(loss_veh)
        records_of_loss_se.append(loss_se)


    # Save trained model weights and architecture, this will be used by the visualization code
    weight_rate = '_' + str(vr_weights) + '_' + str(seenv.datasize_base) + '_' + str(n_vehicles) + '_'
    for vehtype in range(6):
        type = str(vehtype)
        h5file = 'b6_weights30/' + 'after_train'+ weight_rate + type +  ".h5"
        json_file = 'b6_weights30/' + 'after_train' + weight_rate + type +  ".json"
        model_eval_list[vehtype].save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(model_eval_list[vehtype].to_json(), outfile)
        print('file saved: %s, %s' % (h5file, json_file))

    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)

    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    # 保存训练过程中的指标
    parent_path = 'b6_record30/'
    method_name = 'merge_'+weight_rate+'_ds_'+str(seenv.datasize_base)+'_'+str(n_vehicles)+'_'
    print("Saving Loss:")
    with open(parent_path + method_name + "loss_veh.pl", 'wb') as f:
        print("recording loss_veh..")
        pickle.dump(records_of_loss_veh, f)

    with open(parent_path + method_name + "loss_se.pl", 'wb') as f:
        print("recording loss_se..")
        pickle.dump(records_of_loss_se, f)

    with open(parent_path + method_name + "loss_total_cc.pl", 'wb') as f:
        print("recording loss_total..")
        pickle.dump(records_of_loss_total, f)

    print("Saving Reward:")
    with open(parent_path + method_name + "total_reward.pl", 'wb') as f:
        print("recording total_reward..")
        pickle.dump(records_of_total_reward, f)

    with open(parent_path + method_name +"pure_total_reward.pl", 'wb') as f:
        print("recording pure total_reward..")
        pickle.dump(records_of_pure_total_reward, f)

    with open(parent_path + method_name + "veh_reward.pl", 'wb') as f:
        print("recording veh_reward..")
        pickle.dump(records_of_veh_reward, f)

    with open(parent_path + method_name + "se_reward.pl", 'wb') as f:
        print("recording se_reward..")
        pickle.dump(records_of_se_reward, f)

    print("Saving Index:")
    with open(parent_path + method_name + "veh_drive.pl", 'wb') as f:
        print("recording veh_drive..")
        pickle.dump(records_of_veh_drive, f)

    with open(parent_path + method_name + "veh_drive_speed.pl", 'wb') as f:
        print("recording veh_drive_speed..")
        pickle.dump(records_of_veh_drive_speed, f)

    with open(parent_path + method_name + "se_delay.pl", 'wb') as f:
        print("recording se_delay..")
        pickle.dump(records_of_se_delay, f)

    with open(parent_path + method_name + "se_SR.pl", 'wb') as f:
        print("recording se_SR..")
        pickle.dump(records_of_se_SR, f)
    print(seenv.SE_data_list)

if __name__ == '__main__':
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    n_vehicles = 30
    datasize_base = 5
    vr_weights = 0.7
    maze = np.array([
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.],
        [0., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1.]
    ])
    qmaze = Qmaze(maze, n_vehicles, time_spend_matrix_veh)
    print(background_vehs)
    print("time:",qmaze.time_matrix)
    print("O:",qmaze.vehs_og_list)
    print("D:",qmaze.vehs_dest_list)
    seenv = SE_Env(maze, qmaze, datasize_base)
    # dis_list 是 Floyd 算法计算出的各个节点之间最短路径，这个变量将在全局使用，作为评估模型的依据
    dis_list = [dis_hop(qmaze.vehs_og_list[veh], qmaze.vehs_dest_list[veh]) for veh in range(n_vehicles)]
    print("dis_list:", dis_list)

    model_eval_list = [build_model_v2se4() for vehtype in range(6)]
    model_target_list = [build_model_v2se4() for vehtype in range(6)]
    # print(len(model_target_list))
    # print(model_eval_list[5])
    # model_eval = build_model_v2se4()
    # model_target = build_model_v2se4()

    qtrain(model_eval_list, model_target_list,
           perform_vehs=0, n_vehicles=n_vehicles,vr_weights = vr_weights,
           n_epoch=1000, max_memory=8 * maze.size, data_size=32)
    print("\nthe end of train \n")
    pass
