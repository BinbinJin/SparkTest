package com

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/23.
  */
object modelMerge {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("modelMerge").setMaster("local")
    val sc = new SparkContext(conf)
    val per = Array(0.15,0.15,0.15,0.55)
    val input1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\MF\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(0))
    }
    val input2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(1))
    }
    val input3 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD++\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(2))
    }
    val input4 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\FM\\FM_pp.csv").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(3))
    }
    input1.union(input2).union(input3).union(input4)
      .reduceByKey(_+_)
      .map(x=>x._1._1+","+x._1._2+","+x._2)
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\per\\MF_SVD_SVD++_FM")
  }
}
