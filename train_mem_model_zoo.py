from zoo import init_nncontext
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
from data_utils import generate_xym
from AR_mem.config import Config
from AR_mem.model import Model
from time import time
import tensorflow as tf
from zoo.common import set_core_number
from zoo.util.spark import processPDFrame

from bigdl.util.common import get_node_and_core_number

import pickle

if __name__ == "__main__":

    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    model_dir = sys.argv[4]

    # For tuning
    if len(sys.argv) > 5:
        core_num = int(sys.argv[5])
    else:
        core_num = 4
    if len(sys.argv) > 6:
        thread_num = int(sys.argv[6])
    else:
        thread_num = 10

    config = Config()
    config.data_path = "/home/ding/data/skt/npz"
    config.latest_model=False

    feat_cols = config.feat_cols
    n_feat = config.n_feat
    x_size = config.x_size
    y_size = config.y_size
    m_size = config.m_size
    m_days = config.m_days
    m_gaps = config.m_gaps
    scaler_dump_file = config.scaler_dump

    # init or get SparkContext
    sc = init_nncontext()
    
    def pdFrame(pf):
        df = pf.rename(columns={'evt_dtm': 'EVT_DTM', 'rsrp': 'RSRP', 'rsrq': 'RSRQ',
                                'dl_prb_usage_rate': 'DL_PRB_USAGE_RATE', 'sinr': 'SINR',
                                'ue_tx_power': 'UE_TX_POWER', 'phr': 'PHR',
                                'ue_conn_tot_cnt': 'UE_CONN_TOT_CNT', 'cqi': 'CQI'})

        # load scaler
        with open(scaler_dump_file, 'rb') as scaler_dump:
            scaler = pickle.load(scaler_dump)

        # Normalzing
        df[feat_cols] = scaler.transform(df[feat_cols])

        # Generate X, Y, M
        x, y, m = generate_xym(df[feat_cols].to_numpy(), n_feat, x_size,
                               y_size, m_size, m_days, m_gaps)

        X = x.reshape(-1, 10, 8)
        Y = y.reshape(-1, 8)
        M = m.reshape(-1, 77, 8)
        return X, Y, M

    # tuning
    set_core_number(core_num)

    node_num, core_num = get_node_and_core_number()


    # get feature label list for training
    def get_feature_label_list(data_seq):
        X, Y, M = data_seq
        length = X.shape[0]
        return [([X[i], M[i]], Y[i]) for i in range(length)]


    def pdFrame(pf):
        df = pf.rename(columns={'evt_dtm': 'EVT_DTM', 'rsrp': 'RSRP', 'rsrq': 'RSRQ',
                                'dl_prb_usage_rate': 'DL_PRB_USAGE_RATE', 'sinr': 'SINR',
                                'ue_tx_power': 'UE_TX_POWER', 'phr': 'PHR',
                                'ue_conn_tot_cnt': 'UE_CONN_TOT_CNT', 'cqi': 'CQI'})

        # load scaler
        with open(scaler_dump_file, 'rb') as scaler_dump:
            scaler = pickle.load(scaler_dump)

        # Normalzing
        df[feat_cols] = scaler.transform(df[feat_cols])

        # Generate X, Y, M
        x, y, m = generate_xym(df[feat_cols].to_numpy(), n_feat, x_size,
                               y_size, m_size, m_days, m_gaps)

        X = x.reshape(-1, 10, 8)
        Y = y.reshape(-1, 8)
        M = m.reshape(-1, 77, 8)
        return X, Y, M


    t = processPDFrame(sc, data_path, pdFrame) \
        .flatMap(lambda data_seq: get_feature_label_list(data_seq)) \
        .cache()

    train_rdd, val_rdd, test_rdd = t.randomSplit([config.num_cells_train, config.num_cells_valid, config.num_cells_test])

    dataset = TFDataset.from_rdd(train_rdd,
                                 features=[(tf.float32, [10, 8]), (tf.float32, [77, 8])],
                                 labels=(tf.float32, [8]),
                                 batch_size=batch_size,
                                 val_rdd=val_rdd)

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


