spark-submit \
       --master spark://sr576:7077 \
       --class org.apache.spark.ml.example.LORiqiyiDataExample \
       --conf spark.driver.maxResultSize=50g \
       target/spark-vlbfgs-0.1-SNAPSHOT.jar \
       1.0 10 2100000 90000 0.1 
