from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
from data_utils import load_agg_selected_data_mem_train, get_datasets_from_dir, generate_xym
from AR_mem.config import Config
from AR_mem.model import Model
from time import time
import tensorflow as tf
from zoo.common import set_core_number

from bigdl.util.common import get_node_and_core_number

import pickle
import pandas as pd

if __name__ == "__main__":

    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    model_dir = sys.argv[4]

    # For tuning
    if len(sys.argv) > 5:
        core_num = int(sys.argv[5])
    else:
        core_num = 1
    if len(sys.argv) > 6:
        thread_num = int(sys.argv[6])
    else:
        thread_num = 10

    config = Config()
    config.data_path = "/home/ding/data/skt/npz"
    config.latest_model=False

    # init or get SparkContext
    sc = init_nncontext()
    
    # tuning
    set_core_number(core_num)

    # load from local
    def parse_local_csv(file):

        # load scaler
        with open(config.scaler_dump, 'rb') as scaler_dump:
            scaler = pickle.load(scaler_dump)

        # get CELL_NUM from filename
        cell_num = file.split("/")[-1].split('.')[0]

        print("file")
        print(file)
        df = pd.read_csv(file, header=0)

        df['CELL_NUM'] = int(cell_num)
        df = df.rename(columns={'evt_dtm': 'EVT_DTM', 'rsrp': 'RSRP', 'rsrq': 'RSRQ',
                                        'dl_prb_usage_rate': 'DL_PRB_USAGE_RATE', 'sinr': 'SINR',
                                        'ue_tx_power': 'UE_TX_POWER', 'phr': 'PHR',
                                        'ue_conn_tot_cnt': 'UE_CONN_TOT_CNT', 'cqi': 'CQI'})

        # Normalzing
        df[config.feat_cols] = scaler.transform(df[config.feat_cols])

        # Generate X, Y, M
        x, y, m = generate_xym(df[config.feat_cols].to_numpy(), config.n_feat, config.x_size,
                               config.y_size, config.m_size, config.m_days, config.m_gaps)
        X = x.reshape(-1, 10, 8)
        Y = y.reshape(-1, 8)
        M = m.reshape(-1, 77, 8)
        return X, Y, M


    # load from hdfs
    def parse_hdfs_csv(file):
        import pandas as pd
        from pyarrow import csv
        import pyarrow as pa
        fs = pa.hdfs.connect()

        # load scaler
        with open(config.scaler_dump, 'rb') as scaler_dump:
            scaler = pickle.load(scaler_dump)

        # get CELL_NUM from filename
        cell_num = file.split('.')[0]

        with fs.open(file, 'rb') as f:
            df = pd.read_csv(f, header = 0)

            df['CELL_NUM'] = int(cell_num)
            df = df.rename(columns={'evt_dtm': 'EVT_DTM', 'rsrp': 'RSRP', 'rsrq': 'RSRQ',
                                            'dl_prb_usage_rate': 'DL_PRB_USAGE_RATE', 'sinr': 'SINR',
                                            'ue_tx_power': 'UE_TX_POWER', 'phr': 'PHR',
                                            'ue_conn_tot_cnt': 'UE_CONN_TOT_CNT', 'cqi': 'CQI'})

            # Normalzing
            df[config.feat_cols] = scaler.transform(df[config.feat_cols])

            # Generate X, Y, M
            x, y, m = generate_xym(df[config.feat_cols].to_numpy(), config.n_feat, config.x_size,
                                   config.y_size, config.m_size, config.m_days, config.m_gaps)

        return [x, y, m]

    node_num, core_num = get_node_and_core_number()


    # get feature label list for training
    def get_feature_label_list(data_seq):
        X, Y, M = data_seq
        length = X.shape[0]
        return [([X[i], M[i]], Y[i]) for i in range(length)]

    from os import listdir
    # data_paths = [data_path + f for f in listdir(data_path)]

    hadoop = sc._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration()
    path = hadoop.fs.Path('/')

    for f in fs.get(conf).listStatus(path):
        print f.getPath()

    print("hello")

    data_paths = [data_path + f for f in listdir(data_path)]
    t = sc.parallelize(data_paths, node_num)\
        .map(parse_hdfs_csv)\
        .flatMap(lambda data_seq: get_feature_label_list(data_seq))\
        .coalesce(node_num).cache()

    train_rdd, val_rdd, test_rdd = t.randomSplit([config.num_cells_train, config.num_cells_valid, config.num_cells_test])
    # train_rdd, val_rdd, test_rdd = t.randomSplit(
    #     [2, 1, 1])
    dataset = TFDataset.from_rdd(train_rdd,
                                 features=[(tf.float32, [10, 8]), (tf.float32, [77, 8])],
                                 labels=(tf.float32, [8]),
                                 batch_size=1,
                                 val_rdd=val_rdd)

    # create train data
    # train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = \
    #     load_agg_selected_data_mem_train(data_path=config.data_path,
    #                                x_len=config.x_len,
    #                                y_len=config.y_len,
    #                                foresight=config.foresight,
    #                                cell_ids=config.train_cell_ids,
    #                                dev_ratio=config.dev_ratio,
    #                                test_len=config.test_len,
    #                                seed=config.seed)

    # config.batch_size is useless as we force get_datasets_from_dir return the entire data
    # train_X, train_Y, train_M, valid_X, valid_Y, valid_M, _, _, _ =\
    #     get_datasets_from_dir(config.data_path, config.batch_size,
    #                       train_cells=config.num_cells_train,
    #                       valid_cells=config.num_cells_valid,
    #                       test_cells=config.num_cells_test)[0]
    #
    # dataset = TFDataset.from_ndarrays([train_X, train_M, train_Y], batch_size=batch_size,
    #                                   val_tensors=[valid_X, valid_M, valid_Y],)

    model = Model(config, dataset.tensors[0][0], dataset.tensors[0][1], dataset.tensors[1])
    optimizer = TFOptimizer.from_loss(model.loss, Adam(config.lr),
                                      metrics={"rse": model.rse, "smape": model.smape, "mae": model.mae},
                                      model_dir=model_dir,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                                                    intra_op_parallelism_threads=thread_num)
                                      )

    start_time = time()
    optimizer.optimize(end_trigger=MaxEpoch(num_epochs))
    end_time = time()

    print("Elapsed training time {} secs".format(end_time - start_time))


