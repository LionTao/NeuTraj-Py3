import multiprocessing
import pickle

import numpy as np
import traj_dist.distance as tdist


def trajectory_distance(traj_feature_map, traj_keys, distance_type="hausdorff", batch_size=50, processors=30):
    # traj_keys= traj_feature_map.keys()
    trajs = []
    for k in traj_keys:
        traj = []
        for record in traj_feature_map[k]:
            traj.append([record[1], record[2]])
        trajs.append(np.array(traj))

    pool = multiprocessing.Pool(processes=processors)
    # print np.shape(distance)
    batch_number = 0
    for i in range(len(trajs)):
        if (i != 0) & (i % batch_size == 0):
            print(batch_size * batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size * batch_number:i], trajs, distance_type,
                                                         'geolife'))
            batch_number += 1
    pool.close()
    pool.join()


def trajecotry_distance_list(trajs, distance_type="hausdorff", batch_size=50, processors=30, data_name='porto'):
    """

    Args:
        trajs: 轨迹列表
        distance_type: 轨迹距离类型
        batch_size: batch大小
        processors: 进程数
        data_name: 数据集名称

    Returns:

    """
    pool = multiprocessing.Pool(processes=processors)
    batch_number = 0  # batch计数器
    for i in range(len(trajs)):
        # 按照batch往进程池添加任务
        # 在循环到每一个batch最后一个轨迹时提交这个batch的任务
        if (i != 0) & (i % batch_size == 0):
            print(batch_size * batch_number, i)
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size * batch_number:i], trajs, distance_type,
                                                         data_name))
            batch_number += 1
    pool.close()
    pool.join()


def trajectory_distance_batch(i, batch_trjs, trjs, metric_type="hausdorff", data_name='porto') -> None:
    """

    Args:
        i: 当前batch最后一个轨迹的index
        batch_trjs: 轨迹列表
        trjs: 所有轨迹的列表
        metric_type: 轨迹距离类型
        data_name: 数据集名称

    Returns: None

    """
    if metric_type == 'lcss' or metric_type == 'edr':
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
    # elif metric_type=='erp':
    #     trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
    else:
        # 计算两两之间的距离，也就是这个batch里每一个轨迹和所有轨迹之间的距离
        # 应该是(batch,len(trjs))维度
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)
    # 保存当前batch的数据到pkl文件
    pickle.dump(trs_matrix, open('./features/' + data_name + '_' + metric_type + '_distance_' + str(i), 'wb'))
    print('complete: ' + str(i))


def trajectory_distance_combain(trajs_len, batch_size=100, metric_type="hausdorff", data_name='porto'):
    """

    Args:
        trajs_len: 轨迹数量
        batch_size: batch大小, 需要和trajecotry_distance_list执行时的参数保持一致
        metric_type: 轨迹距离类型
        data_name: 数据集类型

    Returns: (trans_len,轨迹总数)的距离矩阵

    """
    # 加载trajecotry_distance_list的计算结果
    distance_list = []
    for i in range(1, trajs_len + 1):
        if (i != 0) & (i % batch_size == 0):
            temp = pickle.load(open('./features/' + data_name + '_' + metric_type + '_distance_' + str(i), "rb"))
            distance_list.append(temp)
            print(distance_list[-1].shape)
    a = distance_list[-1].shape[1]  # len(trjs)也就是trajecotry_distance_list执行时的轨迹数量
    distances = np.array(distance_list)  # 此时维度应该是(9,200,1874)
    print(distances.shape)  # (9, 200, 1874)
    all_dis = distances.reshape((trajs_len, a))  # 减少维度，整合矩阵
    print(all_dis.shape)  # (1800, 1874)
    pickle.dump(all_dis, open('./features/' + data_name + '_' + metric_type + '_distance_all_' + str(trajs_len), 'wb'))
    return all_dis
