import pickle

import numpy as np

import tools.preprocess as preprocess
from tools.distance_compution import trajectory_distance_combain, trajecotry_distance_list


def distance_comp(coor_path, data_name):
    # load出来的数据格式是(经过cell去重的坐标轨迹集合,[],最长经过cell去重的坐标轨迹长度)
    # 所以[0]拿到的就是经过cell去重的坐标轨迹列表，注意这里的index已经不是原始数据集的index了
    traj_coord = pickle.load(open(coor_path, 'rb'))[0]

    # 转换成numpy矩阵列表, 每个轨迹是一个二维矩阵
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array(t))
    print(np_traj_coord[0])
    print(np_traj_coord[1])
    print(len(np_traj_coord))

    distance_type = 'discret_frechet'

    # 计算每一个轨迹和所有轨迹的距离
    trajecotry_distance_list(np_traj_coord, batch_size=200, processors=15, distance_type=distance_type,
                             data_name=data_name)

    trajectory_distance_combain(1800, batch_size=200, metric_type=distance_type, data_name=data_name)


if __name__ == '__main__':
    # 生成经过cell去重的轨迹列表和数据集的名称
    coor_pkl_path, data_name = preprocess.trajectory_feature_generation(path='./data/toy_trajs')
    # 计算距离
    distance_comp(coor_pkl_path, data_name)
