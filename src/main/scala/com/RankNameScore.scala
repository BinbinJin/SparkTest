package com

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/10/24.
  */
object RankNameScore {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
    val sc = new SparkContext(conf)
    val question = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\SVD++\\SVD++\\data\\all\\question.txt").map(_.split(" "))
      .map(x=>(x(1).toInt,x(0))).collect().toMap
    val user = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\SVD++\\SVD++\\data\\all\\user.txt").map(_.split(" "))
      .map(x=>(x(1).toInt,x(0))).collect().toMap

    val test = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\SVD\\SVD\\data\\all\\finalTest.txt").map{x=>
      val info = x.split(" ")
      val uid = info(4).split(":")(0).toInt
      val qid = info(5).split(":")(0).toInt
      val q = question.get(qid).get
      val u = user.get(uid).get
      (q,u)
    }.zipWithIndex().map(_.swap)

    val score = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\SVD\\SVD\\pred.txt").zipWithIndex().map(_.swap)

    test
      .join(score)
      .map(x=>x._2._1._1+","+x._2._1._2+","+x._2._2)
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\SVD\\SVD\\score")

  }
}
