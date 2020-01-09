import copy, random
import numpy as np
import networkx as nx
from env import *
seed = 10
np.random.seed(seed)
random.seed(seed)
# 蜂窝型的地图区域映射到矩形地图上的坐标点
transfer_dict = {
    (2, 2): 0,(1, 2): 1,(2, 3): 2,(3, 3): 3,(3, 2): 4,(2, 1): 5,(1, 1): 6,(0, 2): 7,(1, 3): 8,(2, 4): 9,
    (3, 4): 10,(4, 4): 11,(4, 3): 12,(4, 2): 13,(3, 1): 14,(2, 0): 15,(1, 0): 16,(0, 0): 17,(0, 1): 18
    }
# cells_list => 转换后的二维地图，自矩阵左上角计数(有效区域)的序号，共18个有效区
# 仅用于 actor 的起点和终点的选取
cells_list = list(transfer_dict.keys())

# 邻接矩阵
veh_move = [
    [1, 2, 3, 4, 5, 6],  # 0
    [7, 8, 2, 0, 6, 18],  # 1
    [8, 9, 10, 3, 0, 1],  # 2
    [2, 10, 11, 12, 4, 0],  # 3
    [0, 3, 12, 13, 14, 5],  # 4
    [6, 0, 4, 14, 15, 16],  # 5
    [18, 1, 0, 5, 16, 17],  # 6
    [-1, -1, 8, 1, 18, -1],  # 7
    [-1, -1, 9, 2, 1, 7],  # 8
    [-1, -1, -1, 10, 2, 8],  # 9
    [9, -1, -1, 11, 3, 2],  # 10
    [10, -1, -1, -1, 12, 3],  # 11
    [3, 11, -1, -1, 13, 4],  # 12
    [4, 12, -1, -1, -1, 14],  # 13
    [5, 4, 13, -1, -1, 15],  # 14
    [16, 5, 14, -1, -1, -1],  # 15
    [17, 6, 5, 15, -1, -1],  # 16
    [-1, 18, 6, 16, -1, -1],  # 17
    [-1, 7, 1, 6, 17, -1]
]

# 根据qmaze生成背景车
maze = np.array([
    [1., 1., 1., 0., 0.],
    [1., 1., 1., 1., 0.],
    [1., 1., 1., 1., 1.],
    [0., 1., 1., 1., 1.],
    [0., 0., 1., 1., 1.]
])
background_vehs = np.array([
    [3., 3., 4., 0., 0.],
    [3., 1., 2., 1., 0.],
    [4., 2., 6., 1., 4.],
    [0., 4., 1., 2., 4.],
    [0., 0., 4., 4., 3.]
])
# background_vehs = np.zeros([5, 5], int)
# for i in range(5):
#     for j in range(5):
#         if maze[i][j] != 0:
#             # i -> veh[i][action] = distance
#             background_vehs[i][j] = 1 * random.randint(1, 5)

time_spend_matrix_veh = np.zeros([19, 19], int)
for i in range(19):
    for j in range(6):
        if veh_move[i][j] != -1:
            time_spend_matrix_veh[i][veh_move[i][j]] = 1 * background_vehs[cells_list[i]]


# 邻接距离矩阵，该矩阵目前设置各个邻接点之间的距离为1，留出做复杂路径最短路径规划
AdjacentMatrix = np.zeros([19, 19], int)
for i in range(19):
    for j in range(6):
        if veh_move[i][j] != -1:
            # 目前设置所有的邻接点之间距离为 1
            AdjacentMatrix[i][veh_move[i][j]] = 1
            # *random.randint(4,12)

# 邻接时间矩阵，表示 actor 在两个邻接点之间迁移所需要的时间
# 时间为 0 时表示两点之间不可直达，actor 生成的 action 正常不应该出现这种情况
time_spend_matrix = np.zeros([19, 19], int)
for i in range(19):
    for j in range(6):
        if veh_move[i][j] != -1:
            # i -> veh[i][action] = distance
            time_spend_matrix[i][veh_move[i][j]] = 1 * random.randint(1, 5)

# A是邻接矩阵，相邻对象元素为1，不相邻为0
D = copy.deepcopy(AdjacentMatrix)
time_spend_matrix_D = copy.deepcopy(time_spend_matrix_veh)
# 用于储存节点对的最短路径，相邻的为实际权值（本例为1），不相邻设置为很大的数（远大于所有边的权值，本例设置为999）
ilter = [i for i in range(len(AdjacentMatrix))]
#ilter = [i for i in range(len(time_spend_matrix))]

# o代表起始节点ID,d是终点ID,mid是内插节点
for o in ilter:
    for d in ilter:
        if d == o:
            continue
        if D[o][d] == 0 or D[o][d] == -1:
            D[o][d] = 999
        if time_spend_matrix_D[o][d] == 0 or time_spend_matrix_D[o][d] == -1:
            time_spend_matrix_D[o][d] = 999

# 使用Floyd算法计算SP
for mid in ilter:
    for o in ilter:
        for d in ilter:
            if D[o][mid] != 999 and D[mid][d] != 999 and D[o][d] > D[o][mid] + D[mid][d]:
                D[o][d] = D[o][mid] + D[mid][d]
            if time_spend_matrix_D[o][mid] != 999 and time_spend_matrix_D[mid][d] != 999 and time_spend_matrix_D[o][d] > time_spend_matrix_D[o][mid] + time_spend_matrix_D[mid][d]:
                time_spend_matrix_D[o][d] = time_spend_matrix_D[o][mid] + time_spend_matrix_D[mid][d]

def dis_hop(OG, DEST):
    """
    二维矩阵坐标到蜂窝型标志点的转换，获取两点之间最短路径值
    """
    og = transfer_dict[OG]
    dest = transfer_dict[DEST]
    return D[og][dest]  ### 转换规则

def static_shortPath(OG, DEST):
    """
    二维矩阵坐标到蜂窝型标志点的转换，获取两点之间最短路径值
    """
    og = transfer_dict[OG]
    dest = transfer_dict[DEST]
    return time_spend_matrix_D[og][dest]  ### 转换规则

G = nx.DiGraph()

length = 19
for i in range(length):
    for j in range(length):
        if time_spend_matrix_veh[i][j] != 0:
            G.add_edge(i, j,weight=time_spend_matrix_veh[i][j])

def Dijkstra(start,end):
    RG = G.reverse()
    dist = {}
    previous = {}
    for v in RG.nodes():
        dist[v] = float('inf')
        previous[v] = 'none'
    dist[end] = 0
    u = end
    while u != start:
        u = min(dist, key=dist.get)
        distu = dist[u]
        del dist[u]
        for u,v in RG.edges(u):
            if v in dist:
                alt = distu + RG[u][v]['weight']
                if alt < dist[v]:
                    dist[v] = alt
                    previous[v] = u
    path = (start,)
    last = start
    while last != end:
        nxt = previous[last]
        path += (nxt,)
        last = nxt
    return path[1:]

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


def origin_dest_action(og, de):
    """
    :param og: 地点坐标
    :param de: 终点坐标
    :return: action
    """
    d_x, d_y = og[0] - de[0], og[1] - de[1]
    if d_x == d_y == 0:
        raise Exception("起始点和终点不能一样")
    if d_x == 0:
        if d_y == 1:
            return LEFT
        else:
            return RIGHT
    if d_x == 1:
        if d_y == 1:
            return UPL
        else:
            return UP
    if d_x == -1:
        if d_y == 0:
            return DOWN
        else:
            return DOWNR


if __name__ == '__main__':
    background_vehs = np.array([
        [1., 1., 1., 0., 0.],
        [1., 2., 1., 1., 0.],
        [1., 1., 4., 2., 1.],
        [0., 1., 2., 2., 1.],
        [0., 0., 1., 1., 1.]
    ])
    # print(cells_list)
    print(background_vehs)
    # print(time_spend_matrix_veh)
    #print(D)
    #print(time_spend_matrix_D)
    # print(Dijkstra(0, 0))
    print(Dijkstra(1, 4))
    print(Dijkstra(4, 1))
    print(Dijkstra(2, 5))
    print(Dijkstra(5, 2))
    print(Dijkstra(3, 6))
    print(Dijkstra(6, 3))
    # print(Dijkstra(8,14))
