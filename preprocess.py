import pickle
import pandas as pd
from AR_mem.config import Config
from data_utils import generate_xym

config = Config()

# load from local
def parse_local_csv(file):
    # load scaler
    with open(config.scaler_dump, 'rb') as scaler_dump:
        scaler = pickle.load(scaler_dump)

    # get CELL_NUM from filename
    cell_num = file.split("/")[-1].split('.')[0]

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


# get feature label list for training
def get_feature_label_list(data_seq):
    X, Y, M = data_seq
    length = X.shape[0]
    return [([X[i], M[i]], Y[i]) for i in range(length)]