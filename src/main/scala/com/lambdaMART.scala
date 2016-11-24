package com

import java.io.PrintWriter

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/12.
  */
object lambdaMART {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("lambdaMART").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val dataName = "219+2+6_final"
    lambdaMART(sc,dataName)
    //lambdaMART_cv(sc,dataName)
  }

  def lambdaMART_cv(sc:SparkContext,inputName:String): Unit ={
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
//    sc
//      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\user_info.txt")
//      .map({x=>
//        val info = x.split("\t")
//        val user = info(0)
//        user
//      })
//      .zipWithIndex().saveAsTextFile("F:\\LambdaMART\\RankLib-v2.1\\data\\user")

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
//    sc
//      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\question_info.txt")
//      .map({x=>
//        val info = x.split("\t")
//        val question = info(0)
//        question
//      })
//      .zipWithIndex().saveAsTextFile("F:\\LambdaMART\\RankLib-v2.1\\data\\question")

    val RDDArr = new Array[RDD[String]](8)
    for (i<-0 until 8){
      RDDArr(i) = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\AnswerRate6_DesNum6_intersectionNum4_dis163_uPreferDis33_hot7\\part"+i+"\\part-00000").cache()
    }
    for (i<-0 until 8) {
      var trainInput = sc.makeRDD(new Array[String](0))
      for (j <- 0 until 8 if j != i) {
        trainInput = trainInput.union(RDDArr(j))
      }
      val validationInput = RDDArr(i)
      val train = trainInput
        .map({x=>
          val info = x.split("\t")
          val question = info(0)
          val user = info(1)
          val qNum = questionMap(question).toInt
          val uNum = userMap(user).toInt
          val labelAndFeature = info(2).split(" ")
          val label = labelAndFeature(0).toDouble.toInt
          (label,qNum,uNum)
        })
        .sortBy(_._2)
        .map(x=>x._1 + " qid:" + x._2 + " " + x._3+":1.0")
        .collect()
      val out = new PrintWriter("F:\\LambdaMART\\RankLib-v2.1\\data\\cv\\"+inputName+"\\train"+i+".txt")
      for (s<-train){
        out.println(s)
      }
      out.close()
      val validation = validationInput
        .map({x=>
          val info = x.split("\t")
          val question = info(0)
          val user = info(1)
          val qNum = questionMap(question).toInt
          val uNum = userMap(user).toInt
          val labelAndFeature = info(2).split(" ")
          val label = labelAndFeature(0).toDouble.toInt
          (label,qNum,uNum)
        })
        .sortBy(_._2)
        .map(x=>x._1 + " qid:" + x._2 + " " + x._3+":1.0")
        .collect()

      val out2 = new PrintWriter("F:\\LambdaMART\\RankLib-v2.1\\data\\cv\\"+inputName+"\\validation"+i+".txt")
      for (s<-validation){
        out2.println(s)
      }
      out2.close()

    }
  }

  def lambdaMART(sc:SparkContext,inputName:String): Unit ={
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
//    sc
//      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\user_info.txt")
//      .map({x=>
//        val info = x.split("\t")
//        val user = info(0)
//        user
//      })
//      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\user")

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
//    sc
//      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\question_info.txt")
//      .map({x=>
//        val info = x.split("\t")
//        val question = info(0)
//        question
//      })
//      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\question")

    val sV = sc.textFile("C:\\Users\\zjcxj\\IdeaProjects\\sparkTest\\Singular.txt").map{x=>x.split(" ").map(_.toDouble)}.collect()(0)
    val qLV = sc.textFile("C:\\Users\\zjcxj\\IdeaProjects\\sparkTest\\ItemMat.txt").map{x=>
      val vec = x.split(" ").map(_.toDouble)
      val s = for (i<-vec.indices) yield (i+228)+":"+vec(i) * sV(i)
      s.toArray
    }.zipWithIndex().map(_.swap).collect().toMap
    val uLV = sc.textFile("C:\\Users\\zjcxj\\IdeaProjects\\sparkTest\\UserMat.txt").map{x=>
      val vec = x.split(" ").map(_.toDouble)
      val s = for (i<-vec.indices) yield (i+243)+":"+vec(i) * sV(i)
      s.toArray
    }.zipWithIndex().map(_.swap).collect().toMap


    val trainData = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\train\\part-*")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val qNum = questionMap(question)
        val uNum = userMap(user)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature.head.toDouble.toInt + 1
        val feature = labelAndFeature.tail++qLV(qNum)++uLV(uNum)
        (label,qNum,feature)
      })
      .sortBy(x=>x._2)
      .map(x=>x._1 + " qid:" + x._2 + " " + x._3.mkString(" "))
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\"+inputName+"\\train")

//    val data = trainData.randomSplit(Array(0.875,0.125),11L)
//    data(0).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\"+inputName+"\\train")
//    data(1).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\lambdaMART\\"+inputName+"\\validate")
    val testData = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\finalTest\\part-*")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val qNum = questionMap(question)
        val uNum = userMap(user)
        val feature = info(2)+" "+qLV(qNum).mkString(" ")+" "+uLV(uNum).mkString(" ")
        (qNum,feature,question,user)
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
