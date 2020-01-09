import tensorflow.keras as keras
from keras import Input, Model
from keras.layers import Reshape, Conv2D, Dense, MaxPool1D, concatenate, Flatten
import numpy as np

# compare/mm
def build_model(maze_size=19, num_actions=6):
    # veh_actions
    input1 = Input(shape=(5, 5), name='input1')
    input2 = Input(shape=(5, 5), name='input2')
    input3 = Input(shape=(5, 5), name='input3')
    x1 = Reshape((5, 5, 1))(input1)
    x2 = Reshape((5, 5, 1))(input2)
    x3 = Reshape((5, 5, 1))(input3)
    x = concatenate([x1, x2, x3])
    x = Conv2D(25, (2, 2), activation='relu')(x)
    x = Conv2D(25, (2, 2), activation='relu')(x)
    x = Conv2D(50, (2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(maze_size, activation='relu')(x)
    output = Dense(num_actions, name='output')(x)
    model = Model(inputs=[input1, input2, input3], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_model_se(maze_size=19):
    # SE_action
    # input1 = Input(shape=(5, 5), name='input1')
    # input2 = Input(shape=(5, 5), name='input2')
    input3 = Input(shape=(5, 5), name='input3')
    input4 = Input(shape=(5, 5), name='input4')
    input5 = Input(shape=(5, 5), name='input5')
    # x1 = Reshape((5, 5, 1))(input1)
    # x2 = Reshape((5, 5, 1))(input2)
    x3 = Reshape((5, 5, 1))(input3)
    x4 = Reshape((5, 5, 1))(input4)
    x5 = Reshape((5, 5, 1))(input5)
    x_ = concatenate([x3, x4, x5])
    x_ = Conv2D(25, (2, 2), activation='relu')(x_)
    x_ = Conv2D(50, (2, 2), activation='relu')(x_)
    x_ = Flatten()(x_)
    x_ = Dense(maze_size, activation='relu')(x_)
    output_ = Dense(maze_size, name='output_')(x_)

    model = Model(inputs=[input3, input4, input5], outputs=[output_])
    model.compile(optimizer='adam', loss='mse')
    return model

#veh根据2信道内容决策，se根据4信道输入决策
def build_model_v2se4(maze_size=19, num_actions=6):
    # veh_actions
    input1 = Input(shape=(5, 5), name='input1')
    input2 = Input(shape=(5, 5), name='input2')
    input3 = Input(shape=(5, 5), name='input3')
    input4 = Input(shape=(5, 5), name='input4')
    x1 = Reshape((5, 5, 1))(input1)
    x2 = Reshape((5, 5, 1))(input2)
    x3 = Reshape((5, 5, 1))(input3)
    x4 = Reshape((5, 5, 1))(input4)
    # veh 输出处理
    x = concatenate([x1, x2])
    x = Conv2D(25, (2, 2), name='conv1x', activation='relu')(x)
    x = Conv2D(25, (2, 2), name='conv2x', activation='relu')(x)
    x = Conv2D(50, (2, 2), name='conv3x', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(maze_size, name='DenseX'  , activation='relu')(x)
    output1 = Dense(num_actions, name='output1')(x)

    #se输出处理
    x_ = concatenate([x1, x2, x3, x4])
    x_ = Conv2D(25, (2, 2), name='conv1x_', activation='relu')(x_)
    x_ = Conv2D(25, (2, 2), name='conv2x_', activation='relu')(x_)
    x_ = Conv2D(50, (2, 2), name='conv3x_', activation='relu')(x_)
    x_ = Flatten()(x_)
    x_ = Dense(maze_size, name='DenseX_', activation='relu')(x_)
    output2 = Dense(maze_size, name='output2')(x_)

    model = Model(inputs=[input1, input2, input3, input4], outputs=[output1, output2])
    model.compile(optimizer='adam', loss='mse',loss_weights=[1.0,1.0])
    return model


def build_model_mergeaction(maze_size=19, num_actions=6):
    # SE_actions
    input1 = Input(shape=(5, 5), name='input1')
    input2 = Input(shape=(5, 5), name='input2')
    input3 = Input(shape=(5, 5), name='input3')
    input4 = Input(shape=(5, 5), name='input4')
    x1 = Reshape((5, 5, 1))(input1)
    x2 = Reshape((5, 5, 1))(input2)
    x3 = Reshape((5, 5, 1))(input3)
    x4 = Reshape((5, 5, 1))(input4)
    x = concatenate([x1, x2, x3, x4])
    x = Conv2D(25, (2, 2), activation='relu')(x)
    x = Conv2D(25, (2, 2), activation='relu')(x)
    x = Conv2D(50, (2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(maze_size*num_actions, activation='relu')(x)
    x = Dense(maze_size*num_actions, activation='relu')(x)
    output = Dense(maze_size*num_actions, name='output')(x)

    model = Model(inputs=[input1, input2, input3, input4], outputs=[output])
    model.compile(optimizer='adam', loss='mse')
    return model

# if __name__ == '__main__':
#     np.random.seed(1)
#     m = build_model_v2se4()
#     weight_conv_1, bias_conv_1 = m.get_layer('conv1x').get_weights()
#     print(weight_conv_1[0][0][0], bias_conv_1)
#     # print(m.get_layer('conv1x').get_weights()[0][0][0])
#     inputs1 = 7*np.ones((1,5,5))
#     inputs2 = np.ones((1,5,5))
#     inputs3 = np.ones((1,5,5))
#     inputs4 = 4*np.ones((1,5,5))
#     targets1 = 0.5*np.ones((1,6))
#     targets2 = np.ones((1,19))
#     print("traindata:", inputs1, inputs2, inputs3, inputs4, targets1, targets2)
#     h = m.fit(
#         [inputs1, inputs2, inputs3, inputs4],
#         [targets1, targets2],
#         verbose=1,
#     )
#
#     weight_conv_1, bias_conv_1 = m.get_layer('conv1x').get_weights()
#     print(weight_conv_1[0][0][0], bias_conv_1)