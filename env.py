visited_mark = 0.8  # 汽车访问过的位置的标识数
cur_mark = 2  # 汽车当前位置的标识数
dest_mark = 2.5  # 汽车目的地的标识数
LEFT = 0
UPL = 1
UP = 2
RIGHT = 3
DOWNR = 4
DOWN = 5
UNGO = -1
GOING = -2

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UPL: 'upleft',
    UP: 'up',
    RIGHT: 'right',
    DOWNR: 'downright',
    DOWN: 'down'
}