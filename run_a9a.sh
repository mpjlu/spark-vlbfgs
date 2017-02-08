spark-submit \
       --master spark://sr576:7077 \
       --class org.apache.spark.ml.example.LORRealDataExample \
       --conf spark.driver.maxResultSize=50g \
       target/spark-vlbfgs-0.1-SNAPSHOT.jar \
       1.0 10 50 5000 0.1 
