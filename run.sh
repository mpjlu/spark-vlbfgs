spark-submit \
       --num-executors 10 \
       --executor-cores 2 \
       --master spark://sr576:7077 \
       --class org.apache.spark.ml.example.RosenbrockExample \
       target/spark-vlbfgs-0.1-SNAPSHOT.jar \
       10 2 10000 100 true true true
