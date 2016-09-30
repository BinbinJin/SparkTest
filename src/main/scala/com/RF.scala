package com

import java.io.PrintWriter

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
  * Created by zjcxj on 2016/9/14.
  */
object RF {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("RF").setMaster("local[3]")
    val sc = new SparkContext(conf)
    val dataName = "answerRate_des_intersection_dis_preferDis_hot_RF"
    train(sc,dataName,63)
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
      (label,indices,value,question,user)
    })
    val featureNum = data.flatMap(x=>x._2).max()+1
    val training = data.map(x=>(x._4,x._5,LabeledPoint(x._1,Vectors.sparse(featureNum,x._2,x._3)))).cache()

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

    /*预测*/
//    prediction(training,test,240,12)
    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF\\"+inputName+"_ndcg4.txt")
    for (numTrees <- Range(210,211,10))
      for (maxDepth <- Range(16,17,1)){
        val (pre,ndcg) = localTest(training,numTrees,maxDepth)
        out.println(numTrees+" "+maxDepth+" "+pre+" "+ndcg)
//        prediction(training,test,numTrees,maxDepth)
      }
    out.close()

//    val (pre,ndcg) = localTest(training,200,5)
//    println(pre+" "+ndcg)
  }

  def prediction(train:RDD[(String,String,LabeledPoint)],test:RDD[(String,String,Vector)],numTrees:Int,maxDepth:Int): Unit ={
    val categoricalFeaturesInfo = Map[Int, Int]()
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxBins = 500
    val training = train.map(x=>x._3)

    val model = RandomForest.trainRegressor(training, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    //val model = RandomForestModel.load(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF\\180_8")
    val labelsAndPredictions = test.map { case(qid,uid,point) =>
      val prediction = model.predict(point)
      qid+","+uid+","+prediction
    }.repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF_"+numTrees+"_"+maxDepth +"_sub")
  }

  def localTest(data:RDD[(String,String,LabeledPoint)],numTrees:Int,maxDepth:Int): (Double,Double) ={
    val splits = data.randomSplit(Array(0.9, 0.1),11L)
    val (trainingData, testData) = (splits(0), splits(1))

    val categoricalFeaturesInfo = mutable.HashMap[Int, Int]()
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxBins = 500

    val model = RandomForest.trainRegressor(trainingData.map(_._3), categoricalFeaturesInfo.toMap,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point._3.features)
      (point._1,point._2,prediction,point._3.label.toInt)
    }//.map(x=>x._1+"\t"+x._2+"\t"+x._3+"\t"+x._4).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF100_4")

    val ndcg = NDCG.NDCG(labelsAndPredictions)
    val pre = LR.evaluate(labelsAndPredictions)
    (pre,ndcg)
//    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
//    out.println(numTrees + " " + maxDepth + " " + testMSE)
  }
}
