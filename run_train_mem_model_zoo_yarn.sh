export OMP_NUM_THREADS=10

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master yarn \
 --deploy-mode client \
 --driver-memory 2g \
 --executor-memory 80g \
 --executor-cores 32 \
 --conf spark.driver.host=i01 \
 --num-executors 3 \
 --py-files data_utils.py,${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.8.0-SNAPSHOT-python-api.zip \
 --files scaler.pkl \
train_mem_model_zoo.py hdfs://192.168.111.24:9000/data/byCell/ 1020 1 /home/nvkvs/ding/ARMemNet-jennie-test/model 4
