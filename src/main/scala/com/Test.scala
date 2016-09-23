package com

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by JinBinbin on 2016/8/30.
  */
object Test {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("RF").setMaster("local[3]")
    val sc = new SparkContext(conf)
    val dataName = "15_raw+compress_rawQ_rawU_df0_3rel_userPre50+143_non0"
    dataTrans(sc,dataName,44)
  }
  def dataTrans(sc:SparkContext,inputName:String,modelNum:Int): Unit ={
    val data = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\train_v4.csv").map({x=>
      val info = x.split(",")
      val label = info(5).toDouble
      val feat = new Array[Double](info.length-9)
      for (i<-9 until info.length)
        feat(i-9) = info(i).toDouble
      LabeledPoint(label,Vectors.dense(feat))
    }).cache()

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\pred_v4.csv").map({x=>
      val info = x.split(",")
      val question = info(0)
      val user = info(1)
      val feat = new Array[Double](info.length-9)
      for (i<-9 until info.length) {
        if (info(i)!="") {
          feat(i - 9) = info(i).toDouble
        }
      }
      (question,user,Vectors.dense(feat))
    }).cache()

    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data)
    model.save(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelSub"+modelNum)
    val model2 = LogisticRegressionModel.load(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelSub"+modelNum).clearThreshold()
    val predictionAndLabel = test
      .map({x=>
        val prediction = model2.predict(x._3).toFloat
        x._1+","+x._2+","+prediction
      })
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\resSub"+modelNum)
  }
}
