package com

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/23.
  */
object modelMerge {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("modelMerge").setMaster("local")
    val sc = new SparkContext(conf)
    val input1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\{LR.csv,GBDT.csv}").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score)
    }
      .reduceByKey(_+_)
      .map(x=>x._1._1+","+x._1._2+","+x._2/2)
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\res4")
  }
}
