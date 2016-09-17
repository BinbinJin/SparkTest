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
    val dataName = "15_compress_rawQ_rawU_df7_3rel_userPre50+143"
    //lambdaMART(sc,dataName)
    val invitedInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\label2.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val label = info(2).toInt
        val gbdtFeature = info(3)
        (question,user,label,gbdtFeature)
      }).filter(_._3==0).count()
    println(invitedInfo)
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
    questionId.map(x=>x._1+" "+x._2).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\qid")
    val questionIdMap =questionId.collect().toMap

    val trainData = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\train\\part-*")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val qid = questionIdMap(question)
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
        (label,indices,value,qid)
      })

    val testData = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\test\\part-*")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val qid = questionIdMap(question)
        val user = info(1)
        val feature = info(2).split(" ")
        val indices = new Array[Int](feature.length)
        val value = new Array[Double](feature.length)
        for (i<-0 until feature.length){
          val indAndVal = feature(i).split(":")
          indices(i) = indAndVal(0).toInt-1
          value(i) = indAndVal(1).toDouble
        }
        (indices,value,qid)
      })
    val set1 = trainData.map(x=>x._2.toSet).reduce(_++_)
    val set2 = testData.map(x=>x._1.toSet).reduce(_++_)
    val featureMap = (set1++set2).toArray.zipWithIndex.toMap
println(featureMap.size)

    val train = trainData.map(x=>{
      val indices = x._2
      val newIndices = indices.map(x=>featureMap(x))
      val feature = newIndices.zip(x._3).sortBy(x=>x._1).map(x=>(x._1+1)+":"+x._2)
      (x._1,feature,x._4)
    })
      .sortBy(x=>x._3)
      .map(x=>x._1+" qid:"+x._3+" "+x._2.mkString(" "))
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\train")

    val test = testData.map(x=>{
      val indices = x._1
      val newIndices = indices.map(x=>featureMap(x))
      val feature = newIndices.zip(x._2).sortBy(x=>x._1).map(x=>(x._1+1)+":"+x._2)
      (feature,x._3)
    })
      .sortBy(_._2)
      .map(x=>"0 qid:"+x._2+" "+x._1.mkString(" ")).repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\test")

  }
}
