package com

import java.io.PrintWriter

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
    val dataName = "qid_uid_bin2"
//    dataTrans(sc,dataName,44)
    val count = sc.textFile("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid\\train.txt")
      .map{x=>
        val info = x.split(" ")
        val qid = info(0)
        val uid = info(1)
        val target = if (info(2).toDouble==1.0){1} else {0}
        ((qid,uid),(target,1))
      }
      .reduceByKey((x,y)=>(x._1+y._1,x._2+y._2))
      .map{x=>
        val target = x._2._1*1.0/x._2._2
        val label = if (target == 0){-1} else {target}
        x._1._1+" "+x._1._2+" "+label}
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid\\train_merge")
  }
  def dataTrans(sc:SparkContext,inputName:String,modelNum:Int): Unit ={
    val userMap = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\user_info.txt")
      .map({x=>
        val info = x.split("\t")
        val user = info(0)
        user
      })
      .zipWithIndex()
      .collect()
      .toMap
    sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\user_info.txt")
      .map({x=>
        val info = x.split("\t")
        val user = info(0)
        user
      })
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+inputName+"\\user")

    val questionMap = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\question_info.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        question
      })
      .zipWithIndex()
      .collect()
      .toMap
    sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\question_info.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        question
      })
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+inputName+"\\question")

    val train = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+inputName+"\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      (label,question,user)
    }).collect()

    val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid_cv\\train.txt")
    for ((label,question,user)<-train){
      val row = questionMap(question)
      val col = userMap(user)
      trainFile.println(row+" "+col+" "+label)
    }
    trainFile.close()

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+inputName+"\\test\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      (label,question,user)
    }).collect()

    val testFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid_cv\\validation.txt")
    for ((label,question,user)<-test){
      val row = questionMap(question)
      val col = userMap(user)
      testFile.println(row+" "+col+" "+label)
    }
    testFile.close()
  }
}
