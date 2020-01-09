import numpy as np
# from new_model import build_model

class Experience(object):
    def __init__(self, model_eval, model_target,max_memory=100, discount=0.95):
        self.model_e = model_eval
        self.model_t = model_target
        #self.model_e_s = model_eval_se
        #self. model_t_s = model_target_se
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        # self.num_actions = model_eval.output_shape[-1]
        self.veh_num_actions = model_eval.output_shape[0][-1]
        self.se_num_actions = model_eval.output_shape[1][-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]


    def carry_parameters_totarget(self):
        config = self.model_e.get_weights()
        self.model_t.set_weights(config)

    #def carry_parameters_toperform(self):
    #    config = self.model_e.get_weights()
    #    self.model_p.set_weights(config)


    def predict_e(self, envstate, cell_vehs_num, seenvstate, cell_ses_num):
        return self.model_e.predict([envstate, cell_vehs_num, seenvstate, cell_ses_num])[0][0]
    def predict_t(self, envstate, cell_vehs_num, seenvstate, cell_ses_num):
        return self.model_t.predict([envstate, cell_vehs_num, seenvstate, cell_ses_num])[0][0]
    def predict_e_se(self, envstate, cell_vehs_num, seenvstate, cell_ses_num):
        return self.model_e.predict([envstate, cell_vehs_num, seenvstate,cell_ses_num])[1][0]
    def predict_t_se(self, envstate, cell_vehs_num, seenvstate, cell_ses_num):
        return self.model_t.predict([envstate, cell_vehs_num, seenvstate,cell_ses_num])[1][0]


    def get_data(self, data_size=10):
        #env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs1 = np.zeros((data_size, 5, 5))
        inputs2 = np.zeros((data_size, 5, 5))
        inputs3 = np.zeros((data_size, 5, 5))
        inputs4 = np.zeros((data_size, 5, 5))
        targets1 = np.zeros((data_size, self.veh_num_actions))
        targets2 = np.zeros((data_size, self.se_num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            #envstate, action, reward, envstate_next, game_over, vehsnum_before_act, vehsnum_after_act = self.memory[j]
            envstate, seenvstate, \
            action, seaction, \
            reward, vreward ,sereward,\
            envstate_next, seenvstate_next, \
            game_over, vehsnum_before_act, vehsnum_after_act, \
            sesnum_before_act, sesnum_after_act = self.memory[j]
            inputs1[i] = envstate
            inputs2[i] = vehsnum_before_act
            inputs3[i] = seenvstate
            inputs4[i] = sesnum_before_act
            # There should be no target values for actions not taken.

            targets1[i] = self.predict_e(envstate, vehsnum_before_act, seenvstate, sesnum_before_act)
            a_num = np.argmax(self.predict_e(envstate_next, vehsnum_after_act, seenvstate_next, sesnum_after_act))
            Q_sa_veh = self.predict_t(envstate_next, vehsnum_after_act, seenvstate_next, sesnum_after_act)[a_num]

            targets2[i] = self.predict_e_se(envstate, vehsnum_before_act, seenvstate, sesnum_before_act)
            a_num_se = np.argmax(self.predict_e_se(envstate_next, vehsnum_after_act, seenvstate_next, sesnum_after_act))
            Q_sa_se = self.predict_t_se(envstate_next, vehsnum_after_act, seenvstate_next, sesnum_after_act)[a_num_se]

            Q_sa = (Q_sa_veh + Q_sa_se)/2

           # targets[i] = self.predict_e(envstate)
           # a_num = np.argmax(self.predict_e(envstate_next))
           # Q_sa = self.predict_t(envstate_next)[a_num]
            if game_over:
                targets1[i, int(action)] = reward
                targets2[i, int(action)] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets1[i, int(action)] = reward + self.discount * Q_sa
                targets2[i, int(action)] = reward + self.discount * Q_sa

        return inputs1, inputs2, inputs3, inputs4, targets1, targets2

if __name__ == '__main__':
    pass
    import keras
    m1=build_model()
    m2=build_model()
    experience = Experience(m1,m2)
    #experience.carry_parameters()
    maze = np.array([
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.],
        [0., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1.]
    ])
    maze = maze.reshape((-1,5,5,1))
    #maze = keras.backend.reshape(maze, (-1, 5, 5, 1))
    print(experience.predict_e(maze))