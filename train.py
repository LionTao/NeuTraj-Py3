import os

import tools.config as config
from geo_rnns.neutraj_trainer import NeuTrajTrainer

if __name__ == '__main__':
    # print('os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    # print(config.config_to_str())
    trajrnn = NeuTrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
                             sampling_num=config.sampling_num)
    trajrnn.data_prepare(griddatapath=config.gridxypath, coordatapath=config.corrdatapath,
                         distancepath=config.distancepath, train_radio=config.seeds_radio)

    trajrnn.neutraj_train(load_model=None, in_cell_update=config.incell,
                          stard_LSTM=config.stard_unit)

    # acc1 = trajrnn.trained_model_eval(load_model="model/best_model.h5")
    # print(acc1)
