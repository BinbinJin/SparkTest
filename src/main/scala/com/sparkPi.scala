package com

import scala.math.random
import org.apache.spark.{SparkConf,SparkContext}

/**
  * Created by JinBinbin on 2016/8/29.
  */
object sparkPi {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Spark Pi").setMaster("local")
    val sc = new SparkContext(conf)
    val question = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\data\\cv\\question.txt").map(_.split(" "))
      .map(x=>(x(1).toInt,x(0))).collect().toMap
    val user = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\data\\cv\\user.txt").map(_.split(" "))
      .map(x=>(x(1).toInt,x(0))).collect().toMap

    val test = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\data\\all\\test.txt").map{x=>
      val info = x.split(" ")
      val qid = info(4).split(":")(0).toInt
      val uid = info(5).split(":")(0).toInt
      val q = question.get(qid).get
      val u = user.get(uid).get
      (q,u)
    }.zipWithIndex().map(_.swap)

    val score = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\pred\\pred80.txt").zipWithIndex().map(_.swap)

    test.join(score).map(x=>x._2._1._1+","+x._2._1._2+","+x._2._2).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\score80")

  }
}
