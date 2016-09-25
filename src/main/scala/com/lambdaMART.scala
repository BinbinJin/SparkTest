package com

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/12.
  */
object lambdaMART {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("lambdaMART").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val dataName = "16_raw+compress_3rel_userPre50+143+word2Vec10_20_4"
    lambdaMART(sc,dataName)
  }

  def lambdaMART(sc:SparkContext,inputName:String): Unit ={
    val questionId = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\question_info.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        question
      })
      .zipWithUniqueId()
    questionId.map(x=>x._1+" "+x._2).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\"+inputName+"\\qid")
    val questionIdMap =questionId.collect().toMap

    val trainData = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\train\\part-*")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val qid = questionIdMap(question)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature.head
        val feature = labelAndFeature.tail
        (label,qid,feature)
      })
      .sortBy(x=>x._2)
      .map(x=>x._1 + " qid:" + x._2 + " " + x._3.mkString(" "))
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\"+inputName+"\\train")

    val testData = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\test\\part-*")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val qid = questionIdMap(question)
        val user = info(1)
        val feature = info(2)
        (qid,feature,question,user)
      })
      .sortBy(x=>x._1)
      .repartition(1)

    val scoreSeq = testData
      .map(x=>x._3+","+x._4)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\"+inputName+"\\Seq")
    val test = testData
      .map(x=>"0 qid:"+x._1+" "+x._2)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\"+inputName+"\\test")

  }
}
