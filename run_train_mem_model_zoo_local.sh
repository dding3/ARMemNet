export OMP_NUM_THREADS=10
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master local[1] \
 --driver-memory 20g \
 train_mem_model_zoo.py /home/nvkvs/ding/csv/ 1 1 /home/nvkvs/ARMemNet-jennie/model
