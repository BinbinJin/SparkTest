package com

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/10/28.
  */
object Stacking {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
    val sc = new SparkContext(conf)

    val input1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\MF\\train.txt").distinct().map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2)
      ((qid,uid),score)
    }
    val input2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD\\train.txt").distinct().map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2)
      ((qid,uid),score)
    }
    val input3 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD\\train.txt").map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2)
      ((qid,uid),score)
    }
    val feature = input1.join(input2).join(input3).map(x=>(x._1,(x._2._1._1,x._2._1._2,x._2._2)))

    val invitedInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\invited_info_train.txt")
      .map{x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val label = info(2).toInt
        ((question,user),label)
      }
      .cache()

    val data = invitedInfo.leftOuterJoin(feature).map{x=>
      val label = x._2._1
      val feature1 = x._2._2.get._1
      val feature2 = x._2._2.get._2
      val feature3 = x._2._2.get._3
      val feature = Array(feature1.toDouble,feature2.toDouble,feature3.toDouble)
      LabeledPoint(label,Vectors.dense(feature))
    }

    val input4 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\MF\\test.txt").map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2)
      ((qid,uid),score)
    }
    val input5 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD\\test.txt").map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2)
      ((qid,uid),score)
    }
    val input6 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD\\test.txt").map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2)
      ((qid,uid),score)
    }
    val test = input4.join(input5).join(input6).map{x=>
      val qid = x._1._1
      val uid = x._1._2
      val feature1 = x._2._1._1
      val feature2 = x._2._1._2
      val feature3 = x._2._2
      val feature = Array(feature1.toDouble,feature2.toDouble,feature3.toDouble)
      (qid,uid,Vectors.dense(feature))
    }

    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(data).clearThreshold()
    val preAndLabel = test
      .map{case(qid,uid,feature)=>
        val pre = model.predict(feature)
        qid+","+uid+","+pre
      }.repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\res")
  }
}
