export SPARK_HOME=/root/dingding/spark-2.4.3-bin-hadoop2.7
export ANALYTICS_ZOO_HOME=/root/dingding/Analytics-Zoo
export ENV_HOME=/root/dingding
export OMP_NUM_THREADS=1


PYSPARK_DRIVER_PYTHON=${ENV_HOME}/py37_ding/bin/python PYSPARK_PYTHON=./py37_ding/bin/python ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master yarn \
    --deploy-mode client \
    --executor-memory 20g \
    --driver-memory 10g \
    --executor-cores 1 \
    --num-executors 64 \
    --archives ${ENV_HOME}/py37_ding.tar.gz#py37_ding,${ENV_HOME}/ARMemNet.zip#ARMemNet \
    --conf spark.executorEnv.PYTHONHOME=./py37_ding \
    --conf spark.executorEnv.LD_PRELOAD="./py37_ding/lib/libpython3.7m.so.1.0" \
    --files ./ARMemNet/scaler.pkl \
    ./ARMemNet/train_mem_model_zoo_2.py hdfs://aep-008:9000/data-20k 8192 1 /tmp 1
