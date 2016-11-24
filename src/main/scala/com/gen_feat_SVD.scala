package com

import java.io.PrintWriter

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/10/18.
  */
object gen_feat_SVD {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("gen_feat_SVD").setMaster("local[3]")
    val sc = new SparkContext(conf)
    val dataName = "qid_uid_svd"
    //dataTrans_cv(sc,dataName)
    dataTrans_all(sc,dataName)

  }
  def dataTrans_cv(sc:SparkContext,inputName:String): Unit ={
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
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\cv\\user")

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
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\cv\\question")

    val RDDArr = new Array[RDD[String]](8)
    for (i<-0 until 8){
      RDDArr(i) = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\qid_uid_bin_svd\\part"+i+"\\part-00000").cache()
    }
    for (i<-0 until 8){
      var trainInput = sc.makeRDD(new Array[String](0))
      for (j<-0 until 8 if j!=i) {
        trainInput = trainInput.union(RDDArr(j))
      }
      val validationInput = RDDArr(i)
      val train = trainInput.map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature(0).toDouble
        (label,question,user)
      }).collect()
      val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\cv\\train"+i+".txt")
      for ((label,question,user)<-train){
        val label2 = if(label==1.0){2}else{1}
        val row = questionMap(question)
        val col = userMap(user)
        trainFile.println(label2+" 0 1 1 "+col+":1 "+row+":1")
      }
      trainFile.close()

      val validation = validationInput.map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature(0).toDouble
        (label,question,user)
      }).collect()
      val testFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\cv\\validation"+i+".txt")
      for ((label,question,user)<-validation){
        val label2 = if(label==1.0){2}else{1}
        val row = questionMap(question)
        val col = userMap(user)
        testFile.println(label2+" 0 1 1 "+col+":1 "+row+":1")
      }
      testFile.close()
    }
  }
  def dataTrans_all(sc:SparkContext,inputName:String): Unit ={
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
//      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\all\\user")

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
//      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\all\\question")

//    val train = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
//      val info = x.split("\t")
//      val question = info(0)
//      val user = info(1)
//      val labelAndFeature = info(2).split(" ")
//      val label = labelAndFeature(0).toDouble
//      (label,question,user)
//    }).collect()
//
//    val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\all\\train.txt")
//    for ((label,question,user)<-train){
//      val label2 = if(label==1.0){2}else{1}
//      val row = questionMap(question)
//      val col = userMap(user)
//      trainFile.println(label2+" 0 1 1 "+col+":1 "+row+":1")
//    }
//    trainFile.close()
//
//    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\test\\part-*").map({x=>
//      val info = x.split("\t")
//      val question = info(0)
//      val user = info(1)
//      val labelAndFeature = info(2).split(" ")
//      (question,user)
//    }).collect()
//
//    val testFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data\\all\\test.txt")
//    for ((question,user)<-test){
//      val row = questionMap(question)
//      val col = userMap(user)
//      testFile.println("0 0 1 1 "+col+":1 "+row+":1")
//    }
//    testFile.close()

    val finalTest = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\test_nolabel.txt").map({x=>
      val info = x.split(",")
      val question = info(0)
      val user = info(1)
      (question,user)
    }).collect()

    val finalTestFile = new PrintWriter("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\SVD\\SVD\\data\\all\\finalTest.txt")
    for ((question,user)<-finalTest){
      val row = questionMap(question)
      val col = userMap(user)
      finalTestFile.println("0 0 1 1 "+col+":1 "+row+":1")
    }
    finalTestFile.close()
  }
}
