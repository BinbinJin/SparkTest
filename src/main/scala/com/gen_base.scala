package com

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/11/3.
  */
object gen_base {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("base").setMaster("local")
    val sc = new SparkContext(conf)
    //gen_base(sc)
    test(sc)
  }

  def test(sc:SparkContext): Unit ={
    val res = sc
      .textFile("E:\\SVD++\\merge\\*\\score\\part-00000")
      .map(x=>x.split(","))
      .map(x=>((x(0),x(1)),x(2).toDouble))
      .reduceByKey(_+_).map(x=>x._1._1+","+x._1._2+","+x._2)
      .repartition(1)
      .saveAsTextFile("E:\\SVD++\\merge\\res")
  }

  def gen_base(sc:SparkContext): Unit ={
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
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\user")

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
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\question")
    val train = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val q = questionMap(question)
      val u = userMap(user)
      val labelAndFeature = info(2).split(" ")
      val label = if (labelAndFeature(0).toDouble==1.0){1}else{0}
      u+" "+q+" "+label
    }).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\base")

  }
}
