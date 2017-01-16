/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.example

import org.apache.spark.ml.classification.{LogisticRegression, VLogisticRegression, VLogisticRegressionWithGD}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{Dataset, SparkSession}


object VLORRealDataExample {

  // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a
  def main(args: Array[String]) = {
    val spark = SparkSession
      .builder()
      .appName("VLogistic Regression real data example")
      .getOrCreate()

    val sc = spark.sparkContext

    val dataset1: Dataset[_] = spark.read.format("libsvm").load("data/a9a")
    val dataset2: Dataset[_] = spark.read.format("libsvm").load("data/a9a.t")
    val eva = new MulticlassClassificationEvaluator().setMetricName("accuracy")
/*
    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setRegParam(0.5)
    val model = trainer.fit(dataset1)

    val result = model.transform(dataset2)

    val accu = eva.evaluate(result)
    println("LR accuracy is:" + accu)
    println(s"Logistic regression coefficients: ${model.coefficients}")
*/
    println("args 0:=" + args(0) + "  args 1:=" + args(1))
    val vtrainer = new VLogisticRegressionWithGD()
      .setStepSize(args(0).toDouble)
      .setMaxIter(args(1).toInt)
      .setColsPerBlock(args(2).toInt)
      .setRowsPerBlock(args(3).toInt)
      .setColPartitions(3)
      .setRowPartitions(3)
      .setRegParam(0.5)

    val vmodel = vtrainer.fit(dataset1)

    val vresult = vmodel.transform(dataset2)
    val vaccu = eva.evaluate(vresult)

    println("VLR accuracy is:" + vaccu)
    println(s"VLogistic regression coefficients: ${vmodel.coefficients}")


    sc.stop()
  }
}
