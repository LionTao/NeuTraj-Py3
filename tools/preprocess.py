import pickle as pickle
from typing import List, Tuple

beijing_lat_range = [39.6, 40.7]
beijing_lon_range = [115.9, 117, 1]


class Preprocesser(object):
    def __init__(self, delta=0.005, lat_range=[1, 2], lon_range=[1, 2]):
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self._init_grid_hash_function()

    def _init_grid_hash_function(self):
        dXMax, dXMin, dYMax, dYMin = self.lon_range[1], self.lon_range[0], self.lat_range[1], self.lat_range[0]
        x = self._frange(dXMin, dXMax, self.delta)
        y = self._frange(dYMin, dYMax, self.delta)
        self.x = x
        self.y = y

    def _frange(self, start, end=None, inc=None):
        "A range function, that does accept float increments..."
        if end == None:
            end = start + 0.0
            start = 0.0
        if inc == None:
            inc = 1.0
        L = []
        while 1:
            next = start + len(L) * inc
            if inc > 0 and next >= end:
                break
            elif inc < 0 and next <= end:
                break
            L.append(next)
        return L

    def get_grid_index(self, tuple) -> Tuple[int, int, int]:
        """

        Args:
            tuple: 坐标,(lon,lat)

        Returns: x轴网格坐标，y轴网格坐标，网格编号

        """
        test_tuple = tuple
        test_x, test_y = test_tuple[0], test_tuple[1]
        x_grid = int((test_x - self.lon_range[0]) / self.delta)
        y_grid = int((test_y - self.lat_range[0]) / self.delta)
        index = (y_grid) * (len(self.x)) + x_grid
        return x_grid, y_grid, index

    def traj2grid_seq(self, trajs=[], isCoordinate=False) -> List[List[int]]:
        """

        Args:
            trajs: 坐标轨迹List[List[0,lat,lon]]
            isCoordinate: 是否是坐标，目前都是True

        Returns: 过滤后的坐标点序列，去除了连续在同一个cell里的点, 且只保留满足条件的第一个点
                点格式是(lat,lon)

        """
        grid_traj = []  # 转换成网格轨迹
        for r in trajs:
            # 注意这里的r数据格式为List[0,lat,lon]]
            # 注意get_grid_index接受的是(lon,lat)
            x_grid, y_grid, index = self.get_grid_index((r[2], r[1]))
            # 最终存储的是网格编号
            grid_traj.append(index)

        previous = None
        hash_traj = []
        for index, i in enumerate(grid_traj):
            if previous is None:
                previous = i  # 上一个轨迹cell id
                if not isCoordinate:
                    hash_traj.append(i)
                elif isCoordinate:
                    # 目前只会进入这里
                    # hash_traj第一个是第一个点的坐标
                    hash_traj.append(trajs[index][1:])
            else:
                if i == previous:
                    # 如果是同一个cell就略过
                    pass
                else:
                    if not isCoordinate:
                        hash_traj.append(i)
                    elif isCoordinate:
                        # 如果cell变动了就加进去
                        hash_traj.append(trajs[index][1:])
                    previous = i
        # 总结一下返回值就是去除了连续在同一个cell里的点, 且只保留满足条件的第一个点
        return hash_traj

    def _traj2grid_preprocess(self, traj_feature_map, isCoordinate=False) -> List[List[List[int]]]:
        """

        Args:
            traj_feature_map: 轨迹字典
            isCoordinate: 是否是坐标数据，目前的调用是True

        Returns: 经过cell去重的轨迹字典

        """
        trajs_hash = []
        trajs_keys = traj_feature_map.keys()
        # 对每条轨迹进行基于cell的点去重
        for traj_key in trajs_keys:
            traj = traj_feature_map[traj_key]
            trajs_hash.append(self.traj2grid_seq(traj, isCoordinate))  # (lat,lon)
        return trajs_hash

    def preprocess(self, traj_feature_map, isCoordinate=False):
        """

        Args:
            traj_feature_map: 轨迹字典
            isCoordinate: 是否是网格化后的轨迹，貌似只调用了一次，是True
                            ，此函数传入的数据是坐标数据

        Returns: 得到经过cell去重的坐标轨迹, 有用的轨迹(目前条件下空的), 最大经过cell去重坐标轨迹长度

        """
        if not isCoordinate:
            traj_grids = self._traj2grid_preprocess(traj_feature_map)
            print('gird trajectory nums {}'.format(len(traj_grids)))

            useful_grids = {}
            count = 0
            max_len = 0
            for i, traj in enumerate(traj_grids):
                if len(traj) > max_len:
                    max_len = len(traj)
                count += len(traj)
                for grid in traj:
                    if grid in useful_grids:
                        useful_grids[grid][1] += 1
                    else:
                        useful_grids[grid] = [len(useful_grids) + 1, 1]
            print(len(useful_grids.keys()))
            print(count, max_len)
            return traj_grids, useful_grids, max_len
        elif isCoordinate:
            # 如果是网格化的数据的话进入这里
            # 得到经过cell去重的轨迹 (lat,lon)
            traj_grids = self._traj2grid_preprocess(
                traj_feature_map, isCoordinate=isCoordinate)

            useful_grids = {}  # 这个分支里没用

            # 统计最大长度
            max_len = 0
            for i, traj in enumerate(traj_grids):
                if len(traj) > max_len:
                    max_len = len(traj)
            return traj_grids, useful_grids, max_len


def trajectory_feature_generation(path='./data/toy_trajs',
                                  lat_range: List = beijing_lat_range,
                                  lon_range: List = beijing_lon_range,
                                  min_length=50):
    """

    Args:
        path: 原始数据集路径
        lat_range: 纬度范围
        lon_range: 经度范围
        min_length: 最小原始轨迹长度

    Returns: cell去重坐标轨迹文件名称,数据集名称

    """
    # 拿到文件名称，这里应该是取数据集的名字
    fname: str = path.split('/')[-1].split('_')[0]
    # 原作者提供的toy数据集使用的是python2的pickle所以需要指定encoding
    trajs: List[List[Tuple[float, float]]] = pickle.load(
        open(path, 'rb'), encoding='latin1')

    # 初始化工具类，注意这里的lon & lat range 来自本文件的头部的默认值，及北京的地理数据
    preprocessor = Preprocesser(
        delta=0.001, lat_range=lat_range, lon_range=lon_range)
    print(preprocessor.get_grid_index((lon_range[1], lat_range[1])))

    # 进行轨迹筛选和网格化转换
    max_len = 0  # 最长的网格化后的轨迹长度
    traj_index = {}  # 原始轨迹id和对应的网格化后的轨迹
    for i, traj in enumerate(trajs):
        # traj:List[Tuple[float,float]]
        new_traj = []
        coor_traj = []

        # 只处理满足最短轨迹长度的数据
        if (len(traj) > min_length):
            # 判断轨迹是不是完全在lon & lat range里
            inrange = True
            for p in traj:
                lon, lat = p[0], p[1]
                if not ((lat > lat_range[0]) & (lat < lat_range[1]) & (lon > lon_range[0]) & (lon < lon_range[1])):
                    inrange = False
                new_traj.append([0, p[1], p[0]])  # 原始数据是List[Tuple[lon,lat]],这里转换成了List[List[0,lat,lon]]

            if inrange:
                # 进行基于cell的轨迹点去重
                coor_traj = preprocessor.traj2grid_seq(
                    new_traj, isCoordinate=True)

                if len(coor_traj) == 0:
                    print(len(coor_traj))

                # 只有裁剪后序列在(10,150)长度的轨迹才能幸存
                if ((len(coor_traj) > 10) & (len(coor_traj) < 150)):
                    if len(traj) > max_len:
                        max_len = len(traj)  # 更新max_len,注意这里的max_len是网格化后的轨迹长度
                    traj_index[i] = new_traj

        # 进度条
        if i % 200 == 0:
            print(coor_traj)
            print(i, len(traj_index.keys()))

    print(max_len)
    print(len(traj_index.keys()))

    # 存一下网格轨迹数据
    pickle.dump(traj_index, open(
        './features/{}_traj_index'.format(fname), 'wb'))

    # 点格式是(lat,lon)
    trajs, useful_grids, max_len = preprocessor.preprocess(
        traj_index, isCoordinate=True)

    print(trajs[0])  # 简答看看第一个轨迹

    # 保存 (经过cell去重的坐标轨迹集合,[],最长经过cell去重的坐标轨迹长度) 点格式(lat,lon)
    pickle.dump((trajs, [], max_len), open(
        './features/{}_traj_coord'.format(fname), 'wb'))

    # 找现有数据集中所有轨迹所占区域网格的编号边界
    min_x, min_y, max_x, max_y = 2000, 2000, 0, 0
    for i in trajs:
        # i代表一条经过cell去重的坐标轨迹
        for j in i:
            # j是坐标轨迹中的一个点(lat,lon),所以下面这行要换顺序
            x, y, index = preprocessor.get_grid_index((j[1], j[0]))
            # x轴网格坐标，y轴网格坐标，网格编号

            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
    print(min_x, min_y, max_x, max_y)

    # 进行区域所见, 将原本的geo fence shrink到真实轨迹所占有的网格范围
    all_trajs_grids_xy = []
    for i in trajs:
        # i代表一条经过cell去重的坐标轨迹
        traj_grid_xy = []
        for j in i:
            # j是坐标轨迹中的一个点(lat,lon),所以下面这行要换顺序
            x, y, index = preprocessor.get_grid_index((j[1], j[0]))
            x = x - min_x  # 计算x偏移
            y = y - min_y  # 计算y偏移
            grids_xy = [y, x]  # 生成新的grid id
            traj_grid_xy.append(grids_xy)  # 存储新的grid id
        # 存储新的轨迹 grid id 序列
        all_trajs_grids_xy.append(traj_grid_xy)
    print(all_trajs_grids_xy[0])
    print(len(all_trajs_grids_xy))
    print(all_trajs_grids_xy[0])
    # 保存 (裁剪后的grid id序列集合,[],最长经过cell去重的坐标轨迹长度)
    pickle.dump((all_trajs_grids_xy, [], max_len), open(
        './features/{}_traj_grid'.format(fname), 'wb'))

    # 最终返回的是cell去重坐标轨迹文件名称和数据集名称
    return './features/{}_traj_coord'.format(fname), fname
