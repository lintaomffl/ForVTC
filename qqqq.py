if __name__ == '__main__':
    import numpy as np
    list = np.arange(30)
    res = []
    for i in range(30):
        res.append(list[i] // 5)
    print(res)