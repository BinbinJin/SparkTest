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
    val dataName = "qid_uid"
//    train(sc,dataName,63)
    val numTrees = 240
    val maxDepth = 12
    val (pre,ndcg) = localTest(sc,dataName,numTrees,maxDepth)
    println(pre+"\t"+ndcg)
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
    prediction(training,test,240,12)
//    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\RF\\"+inputName+"_ndcg4.txt")
//    for (numTrees <- Range(210,211,10))
//      for (maxDepth <- Range(16,17,1)){
//        val (pre,ndcg) = localTest(training,numTrees,maxDepth)
//        out.println(numTrees+" "+maxDepth+" "+pre+" "+ndcg)
////        prediction(training,test,numTrees,maxDepth)
//      }
//    out.close()

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

  def localTest(sc:SparkContext,inputName:String,numTrees:Int,maxDepth:Int): (Double,Double) ={
    val data = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+inputName+"\\train\\part-*").map({x=>
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
    val train = data.map(x=>LabeledPoint(x._1,Vectors.sparse(featureNum,x._2,x._3))).cache()

    val data2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+inputName+"\\test\\part-*").map({x=>
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
    val validaion = data2.map(x=>(x._4,x._5,LabeledPoint(x._1,Vectors.sparse(featureNum,x._2,x._3)))).cache()

    val categoricalFeaturesInfo = mutable.HashMap[Int, Int]()
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxBins = 500

    val model = RandomForest.trainRegressor(train, categoricalFeaturesInfo.toMap,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val labelsAndPredictions = validaion.map { point =>
      val prediction = model.predict(point._3.features)
      (point._1,point._2,prediction,point._3.label.toInt)
    }

    val ndcg = NDCG.NDCG(labelsAndPredictions)
    val pre = LR.evaluate(labelsAndPredictions)
    (pre,ndcg)

  }
}
