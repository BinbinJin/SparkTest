package com

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/11/14.
  */
object finalTest {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("finalTest").setMaster("local")
    val sc = new SparkContext(conf)
    val question = sc.textFile("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid_cv\\question.txt").map{x=>
      val info = x.split("\t")
      val qid = info(0)
      val no = info(1).toInt
      (qid,no)
    }.collect().toMap

    val user = sc.textFile("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid_cv\\user.txt").map{x=>
      val info = x.split("\t")
      val uid = info(0)
      val no = info(1).toInt
      (uid,no)
    }.collect().toMap

    val finalTest = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\validate_nolabel.txt").map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val qNum = question(qid)
      val uNum = user(uid)
      qNum+" "+uNum+" 0"
    }.repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid\\validate")

    sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\validate_nolabel.txt").map{x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      qid+" "+uid+" 0"
    }.repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\libmf-2.01\\data\\qid_uid\\validate_qid_uid")
  }
}
