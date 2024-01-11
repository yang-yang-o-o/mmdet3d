import numpy as np


class KMMatcher:
    # 在KM算法中，目的是找到一个完美匹配，使得总成本最小（或总利益最大）
    def __init__(self):
        """
        Initialization Function

        """
        self.matched = None # 存储匹配结果，matched[i] 表示 Y 集合中下标i的元素的匹配情况（X集合中的下标）
        self.info_matrix = None # 用于存储信息矩阵，即成本或利益矩阵。这是一个二维数组，其元素表示X集合和Y集合之间的关联成本或利益（IOU或者其他距离）。
        self.label_x = None # 记录X，Y集合中元素的权值
        self.label_y = None
        self.vis_x = None # 记录X，Y集合元素被访问情况
        self.vis_y = None
        self.slack = None # 存储Y集合中每个元素与X集合中未匹配元素的最小边权差。这是KM算法优化搜索过程的关键部分。
        self.transposed = False # 是否对信息矩阵进行转置
        self.num_x = 0 # 存储X集合和Y集合中的元素数量
        self.num_y = 0

    def setInformationMatrix(self, matrix):
        """
        Interface for setting information matrix

        """
        matrix = np.array(matrix)
        self.info_matrix = matrix
        self.num_x = matrix.shape[0]
        self.num_y = matrix.shape[1]
        self.transposed = False

        if self.num_x > self.num_y: # 如果X集合的元素多于Y集合，对信息矩阵取转置，因为算法给每个X都找到一个匹配的Y，X的数量需要小于等于Y的数量
            self.transposed = True
            self.info_matrix = self.info_matrix.transpose()
            matrix = matrix.transpose()
            tmp = self.num_y
            self.num_y = self.num_x
            self.num_x = tmp

        # print(self.transposed)
        self.label_x = np.full((matrix.shape[0]), float("-inf"))
        for i in range(self.num_x): # info_matrix 里每行的 最大值 放到 label_x
            for j in range(self.num_y):
                if self.label_x[i] < self.info_matrix[i][j]:
                    self.label_x[i] = self.info_matrix[i][j]
        self.label_y = np.zeros((matrix.shape[1])) # label_y 设为全 0

        self.vis_x = np.zeros_like(self.label_x) # 全0
        self.vis_y = np.zeros_like(self.label_y) # 全0

        self.matched = np.full((matrix.shape[1]), -1) # 和label_y相同维度，全-1，存储每个y匹配的x
        self.slack = np.zeros(matrix.shape[1]) # 和label_y相同维度，全0，
        # self.slack 帮助算法追踪和更新当前节点到所有未匹配节点的最小边权差，从而使得算法能够有效地找到改进匹配的机会
        # self.slack 在KM算法中扮演了优化搜索和提高匹配效率的关键角色。通过跟踪和更新这个数组，算法能够更高效地找到成本最优的匹配方案。

    def __perfectMatch(self, x):
        """
        Finding perfect matching in its equal sub-graph

        对于当前的x, 遍历所有没有访问过(delta==0)的y, 求一个边权差delta, 
        如果delta等于0, 
            如果这个y没有被匹配过, 那么当前的x就匹配这个y, 返回true。
            如果这个y被匹配过了, 去递归看看, 已经匹配这个y的那个x能不能找到新的完美匹配,
                如果能找到(这里包含了一个递归链, 不断向上递归找), 那个x就匹配新的y, 当前的这个y就给当前的这个x来匹配。
                如果那个x找不到新的完美匹配, 那么当前的这个x就不能匹配当前的这个y, 就只能返回一个false, 然后取重新调整元素的权值(±delta)
        如果delta不等于0, 那么就维护一个最小的delta, 然后返回false。
                
        """
        self.vis_x[x] = True
        for y in range(self.num_y):
            if self.vis_y[y]:
                continue

            delta = self.label_x[x] + self.label_y[y] - self.info_matrix[x][y]
            if abs(delta) < 1e-3: # delta等于0
                self.vis_y[y] = True
                if self.matched[y] == -1 or self.__perfectMatch(self.matched[y]): # 如果y没有被匹配，或者已经和y建立的匹配不是完美匹配
                    self.matched[y] = x # 更新匹配
                    return True # 对于x，找到完美匹配
            elif self.slack[y] > delta: # 存储Y集合中的每个节点y与X集合中未匹配节点的最小边权差
                self.slack[y] = delta

        return False

    def processKM(self):
        """
        Processing Kuhn Munkres matching algorithm

        遍历每个x, 为每个x都找到一个完美匹配
            在while 循环中不断去找, 找到了以后, break, 然后为下一个x找完美匹配
                如果当前次寻找, 没有为当前的x找到一个完美匹配,
                    那么就更新(±delta) slack 数组 和 X,Y集合中元素的权值(label_x和label_y), 然后再一次为当前的x找完美匹配
                        delta 为 当前没有被访问过的y和X集合中未匹配节点的最小边权差 的最小值
        """
        for x in range(self.num_x):
            self.slack = np.full((self.num_y), float("+inf"))
            while True:
                self.vis_x = np.zeros(self.num_x)
                self.vis_y = np.zeros(self.num_y)
                if self.__perfectMatch(x):
                    break
                else:
                    idx_array_nvisy = [i for i in range(
                        self.num_y) if not self.vis_y[i]]
                    idx_array_visy = [i for i in range(
                        self.num_y) if self.vis_y[i]]
                    idx_array_visx = [i for i in range(
                        self.num_x) if self.vis_x[i]]

                    if len(idx_array_nvisy) != 0:
                        delta = np.min(self.slack[idx_array_nvisy])
                        self.slack[idx_array_nvisy] -= delta
                    if len(idx_array_visx) != 0:
                        self.label_x[idx_array_visx] -= delta
                    if len(idx_array_visy) != 0:
                        self.label_y[idx_array_visy] += delta

        return True

    def getMatchedResult(self):
        """
        Getting KM algorithm matched results

        返回 matched 数组, 如果信息矩阵被转置过, 那么返回“转置”后的 matched 数组

        最终 matched[i] 表示 集合Y 中下标为i的元素 匹配了 集合X 中下标为 matched[i] 的元素

        """
        if not self.transposed:
            return self.matched.astype(np.int16)
        else:
            tmp_matched = np.zeros((self.num_x))
            for i in range(self.matched.size):
                idx = self.matched[i]
                if idx == -1:
                    continue
                tmp_matched[idx] = i
            return tmp_matched.astype(np.int16)

# 参考：
#       [km算法入门](https://www.cnblogs.com/logosG/p/logos.html)
#       [GPT4对于当前.py文件的解释]