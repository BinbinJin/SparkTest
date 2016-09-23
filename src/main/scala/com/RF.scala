package com

import java.io.PrintWriter

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/14.
  */
object RF {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("RF").setMaster("local[3]")
    val sc = new SparkContext(conf)
    val dataName = "15_compress_rawQ_rawU_df0_3rel_userPre50+143_non0_L1"
    train(sc,dataName,43)
  }
  def train(sc:SparkContext,inputName:String,modelNum:Int): Unit ={
    val data = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      val indices = new Array[Int](labelAndFeature.length-1)
      val value = new Array[Double](labelAndFeature.length-1)
      for (i<-1 until labelAndFeature.length){
        val indAndVal = labelAndFeature(i).split(":")
        indices(i-1) = indAndVal(0).toInt-1
        value(i-1) = indAndVal(1).toDouble
      }
      (label,indices,value)
    })
    val featureNum = data.flatMap(x=>x._2).max()+1
    val training = data.map(x=>LabeledPoint(x._1,Vectors.sparse(featureNum,x._2,x._3))).cache()

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\test\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val feature = info(2).split(" ")
      val indices = new Array[Int](feature.length)
      val value = new Array[Double](feature.length)
      for (i<-0 until feature.length){
        val indAndVal = feature(i).split(":")
        indices(i) = indAndVal(0).toInt-1
        value(i) = indAndVal(1).toDouble
      }
      (question,user,Vectors.sparse(featureNum,indices,value))
    }).cache()
    // Split the data into training and test sets (30% held out for testing)
    val splits = training.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 200 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
   val maxDepth = 5
    val maxBins = 100

//    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF\\sta.txt")
//    for (numTrees<-Range(200,270,20);maxDepth<-4 to 8){
//      val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
//        numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
//
//      // Evaluate model on test instances and compute test error
//      model.save(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF\\"+numTrees+"_"+maxDepth)
//      val labelsAndPredictions = testData.map { point =>
//        val prediction = model.predict(point.features)
//        (point.label, prediction)
//      }
//      //labelsAndPredictions.saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF200_5")
//      val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
//      out.println(numTrees + " " + maxDepth + " " + testMSE)
//    }
//    out.close()

    val model = RandomForest.trainRegressor(training, categoricalFeaturesInfo,
            numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    //val model = RandomForestModel.load(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF\\180_8")
    val labelsAndPredictions = test.map { case(qid,uid,point) =>
              val prediction = model.predict(point)
              qid+","+uid+","+prediction
            }.repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF200_5_487_sub")
 //   labelsAndPredictions.saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF200_5")

    //println("Learned regression forest model:\n" + model.toDebugString)
  }
}
