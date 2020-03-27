export SPARK_HOME=/root/dingding/spark-2.4.3-bin-hadoop2.7
export ANALYTICS_ZOO_HOME=/root/dingding/Analytics-Zoo-Local


${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master local[4] \
    --executor-memory 10g \
    --driver-memory 10g \
    --files ./scaler.pkl \
    train_mem_model_zoo_2.py hdfs://aep-008:9000/raw_csv 1056 1 /tmp 4
