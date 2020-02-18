# AR_mem_config


class Config(object):
    def __init__(self):
        # model params
        self.model = "AR_mem"
        self.nsteps = 10   # equivalent to x_len
        self.msteps = 7
        self.attention_size = 16
        self.l2_lambda = 1e-3
        self.ar_lambda = 0.1    
        self.ar_g = 1           
        
        # data params
        # self.data_path = '../data/aggregated_data_5min_scaled.csv'
        self.data_path = '../data-290k-preprocessed'
        self.nfeatures = 8  # number of col_list in "../config_preprocess.py"
        self.x_len = self.nsteps
        self.y_len = 1
        self.foresight = 0 
        self.dev_ratio = 0.02
#         self.test_len = 7
        self.test_len = 0 
        self.seed = None
        
        # train & test params
        # self.train_cell_ids = list(range(20))  # order of cell_id in "../config_preprocess.py"
        # self.test_cell_ids = []           # order of cell_id in "../config_preprocess.py"
        self.num_cells_train = 1000       # # of cells to train
        self.num_cells_valid = 10         # # of cells to validate
        self.num_cells_test  = 10         # # of cells to test
        self.model_dir = None       # Model directory to use in test mode. For example, "model_save/20190405-05"
        self.latest_model = True    # Use lately saved model in test mode. If latest_model=True, model_dir option will be ignored
        
        # training params
        self.lr = 1e-3
        self.num_epochs = 1000
        self.batch_size = 8192
        self.dropout = 0.8     
        self.nepoch_no_improv = 5
        self.clip = 5
        self.allow_gpu = True
        self.desc = self._desc()

        self.scaler_dump = '/home/nvkvs/ding/scaler.pkl'
        self.feat_cols = ['RSRP', 'RSRQ', 'DL_PRB_USAGE_RATE', 'SINR',
                          'UE_TX_POWER', 'PHR', 'UE_CONN_TOT_CNT', 'CQI']
        self.n_feat = len(self.feat_cols)  # features per item

        # preprocess properties
        self.x_size = 10        # # of items per X
        self.m_size = 11        # # of items per M
        self.y_size = 1         # # of items per Y
        self.m_days = 7         # # of days per M
        self.m_gaps = 12*24     # # of rows per day - using 60min / 5min/row x 24hours


    def _desc(self):
        desc = ""
        for mem, val in self.__dict__.items():
            desc += mem + ":" + str(val) + ", "
        return desc


if __name__ == "__main__":
    config = Config()
    print(config.desc)

