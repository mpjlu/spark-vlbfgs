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

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.ml.classification.{LogisticRegression, VLogisticRegression, VLogisticRegressionWithGD}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Dataset, Row}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint

object VLORiqiyiDataExample {

  // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a
  def main(args: Array[String]) = {
    val spark = SparkSession
      .builder()
      .appName("VLogistic Regression real data example")
      .getOrCreate()

    val sc = spark.sparkContext

    val rawData = sc.textFile("data/iqiyi_ios", 70).map(_.split("\\s")).map(x => {
      if (x(0).toInt > 3)
        x(0) = "1"
      else
        x(0) = "0"
      val v: Array[(Int, Double)] = x.drop(1).map(_.split(":"))
        .map(x => (x(0).toInt - 1, x(1).toDouble))
        .sortBy(_._1)
      (x(0).toInt, v)
    }).repartition(100)

    val length = rawData.map(_._2.last._1).max + 1
    println("length: " + length)
    val training = rawData.map{case(label, v) => LabeledPoint(label, Vectors.sparse(length, v.map(_._1), v.map(_._2)))}
    training.cache()
    
    import spark.implicits._
    //val iqiyi_dataset = spark.createDataset(training)
    val iqiyi_dataset = training.toDS()

    println("args 0:=" + args(0) + "  args 1:=" + args(1))

    val eva = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val vtrainer = new VLogisticRegressionWithGD()
      .setStepSize(args(0).toDouble)
      .setMaxIter(args(1).toInt)
      .setColsPerBlock(args(2).toInt)
      .setRowsPerBlock(args(3).toInt)
      .setColPartitions(3)
      .setRowPartitions(3)
      .setRegParam(args(4).toDouble)

    val vmodel = vtrainer.fit(iqiyi_dataset.as[LabeledPoint])
/*
    val vresult = vmodel.transform(dataset2)
    val vaccu = eva.evaluate(vresult)

    val vrdd = vresult.select(col("prediction"), col("label").cast(DoubleType)).rdd.map {
      case Row(prediction: Double, label: Double) => (prediction, label)
    }

    val vnum1 = vrdd.filter(_._1 == 1).count()
    val vnum0 = vrdd.count() - vnum1
    println("VLR predicted number of 1:" + vnum1 + "number of 0: " + vnum0)
    println("VLR accuracy is:" + vaccu)
    println(s"VLogistic regression coefficients: ${vmodel.coefficients}")
*/

    sc.stop()
  }
}
