spark-submit \
       --num-executors 140 \
       --executor-cores 2 \
       --master spark://sr576:7077 \
       --class org.apache.spark.ml.example.VLORRealDataExample \
       target/spark-vlbfgs-0.1-SNAPSHOT.jar \
       1.0 10 100 5000 0.22 
