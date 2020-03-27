from zoo import init_nncontext
from zoo import init_spark_on_yarn
from zoo.tfpark import TFOptimizer, TFDataset
from zoo.feature.common import FeatureSet
from bigdl.optim.optimizer import *
from AR_mem.config import Config
from AR_mem.model import Model
import tensorflow as tf

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

    sc = init_nncontext("train mem zoo")

    feature_set = FeatureSet.csv_dataset(data_path, "./ARMemNet")

    dataset = TFDataset.from_feature_set(feature_set,
                                          features=[(tf.float32, [10, 8]), (tf.float32, [77, 8])],
                                          labels=(tf.float32, [8]),
                                          batch_size=batch_size)


    config = Config()
    model = Model(config, dataset.tensors[0][0], dataset.tensors[0][1], dataset.tensors[1])
    optimizer = TFOptimizer.from_loss(model.loss, Adam(config.lr),
                                      metrics={"rse": model.rse, "smape": model.smape, "mae": model.mae},
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                                                    intra_op_parallelism_threads=10)
                                      )

    optimizer.optimize(end_trigger=MaxEpoch(1))
