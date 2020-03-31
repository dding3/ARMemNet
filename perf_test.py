import os
from zoo import init_nncontext
from AR_mem.config import Config
import pickle
from zoo.util.spark import processPDFrame
from data_utils import generate_xym
from preprocess import parse_csv2
import sys
import time

data_path = sys.argv[1]
sc = init_nncontext()
config = Config()

scaler_dump_file = config.scaler_dump
feat_cols = config.feat_cols
n_feat = config.n_feat
x_size = config.x_size
y_size = config.y_size
m_size = config.m_size
m_days = config.m_days
m_gaps = config.m_gaps


start = time.time()
from os import listdir
data_paths = [data_path + f for f in listdir(data_path)]

rdd = sc.parallelize(data_paths, 1)\
    .mapPartitions(parse_csv2).cache()
rdd.count()
end = time.time()
print("Elapse time:")
print(end-start)
