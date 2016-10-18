package com

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/23.
  */
object modelMerge {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("modelMerge").setMaster("local")
    val sc = new SparkContext(conf)
    val per = Array(0.2,0.8)
    val input1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\MF.csv").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(0))
    }
    val input2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FM_pp.csv").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score*per(1))
    }
    input1.union(input2)
      .reduceByKey(_+_)
      .map(x=>x._1._1+","+x._1._2+","+x._2)
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\MF_FM_pp_0.2")
  }
}
