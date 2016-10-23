package com

import java.io.PrintWriter

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/10/21.
  */
object gen_feat_RankSVD {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("RankSVD").setMaster("local[3]")
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
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\user")

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
      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\question")

    val q_u = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      val qNum = questionMap(question)
      val uNum = userMap(user)
      (label,qNum,uNum)
    }).filter(_._1==1.0).map{x=>
      (x._2,x._3)
    }.groupByKey().collect().toMap

    val train = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      (label,question,user)
    }).sortBy(_._2).collect()

    val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\train.txt")
    for ((label,question,user)<-train){
      val label2 = if(label==1.0){1}else{0}
      val row = questionMap(question)
      val col = userMap(user)
      trainFile.println(label2+" 0 1 1 "+row+":1 "+col+":1")
    }
    trainFile.close()
    val trainFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\train_feed.txt")
    var last = questionMap(train(0)._2)
    var num = 1
    for (i<-1 until train.length){
      val row = questionMap(train(i)._2)
      if (row == last){
        num += 1
      }else{
        val uNum = q_u.get(last)
        if (uNum == None){
          trainFeedBack.println(num+" 0")
        }else{
          trainFeedBack.print(num+" ")
          val uArr = uNum.get.toArray.sorted
          val size = uArr.length
          trainFeedBack.print(size+" ")
          val value = 1.0/Math.sqrt(size)
          trainFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
        }
        last = row
        num = 1
      }
    }
    val uNum = q_u.get(last)
    if (uNum == None){
      trainFeedBack.println(num+" 0")
    }else{
      trainFeedBack.print(num+" ")
      val uArr = uNum.get.toArray.sorted
      val size = uArr.length
      trainFeedBack.print(size+" ")
      val value = 1.0/Math.sqrt(size)
      trainFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
    }
    trainFeedBack.close()

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\qid_uid_bin\\test\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      (label,question,user)
    }).sortBy(_._2).collect()

    val testFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\validation.txt")
    for ((label,question,user)<-test){
      val label2 = if(label==1.0){1}else{0}
      val row = questionMap(question)
      val col = userMap(user)
      testFile.println(label2+" 0 1 1 "+row+":1 "+col+":1")
    }
    testFile.close()
    val testFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\validation_feed.txt")
    last = questionMap(test(0)._2)
    num = 1
    for (i<-1 until test.length){
      val row = questionMap(test(i)._2)
      if (row == last){
        num += 1
      }else{
        val uNum = q_u.get(last)
        if (uNum == None){
          testFeedBack.println(num+" 0")
        }else{
          testFeedBack.print(num+" ")
          val uArr = uNum.get.toArray.sorted
          val size = uArr.length
          testFeedBack.print(size+" ")
          val value = 1.0/Math.sqrt(size)
          testFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
        }
        last = row
        num = 1
      }
    }
    val uNum2 = q_u.get(last)
    if (uNum2 == None){
      testFeedBack.println(num+" 0")
    }else{
      testFeedBack.print(num+" ")
      val uArr = uNum2.get.toArray.sorted
      val size = uArr.length
      testFeedBack.print(size+" ")
      val value = 1.0/Math.sqrt(size)
      testFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
    }
    testFeedBack.close()
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

    val q_u = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      val qNum = questionMap(question)
      val uNum = userMap(user)
      (label,qNum,uNum)
    }).filter(_._1==1.0).map{x=>
      (x._2,x._3)
    }.groupByKey().collect().toMap

    val train = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      (label,question,user)
    }).sortBy(_._2).collect()

    val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\train.txt")
    for ((label,question,user)<-train){
      val label2 = if(label==1.0){1}else{0}
      val row = questionMap(question)
      val col = userMap(user)
      trainFile.println(label2+" 0 1 1 "+row+":1 "+col+":1")
    }
    trainFile.close()
    val trainFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\train_feed.txt")
    var last = questionMap(train(0)._2)
    var num = 1
    for (i<-1 until train.length){
      val row = questionMap(train(i)._2)
      if (row == last){
        num += 1
      }else{
        val uNum = q_u.get(last)
        if (uNum == None){
          trainFeedBack.println(num+" 0")
        }else{
          trainFeedBack.print(num+" ")
          val uArr = uNum.get.toArray.sorted
          val size = uArr.length
          trainFeedBack.print(size+" ")
          val value = 1.0/Math.sqrt(size)
          trainFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
        }
        last = row
        num = 1
      }
    }
    val uNum = q_u.get(last)
    if (uNum == None){
      trainFeedBack.println(num+" 0")
    }else{
      trainFeedBack.print(num+" ")
      val uArr = uNum.get.toArray.sorted
      val size = uArr.length
      trainFeedBack.print(size+" ")
      val value = 1.0/Math.sqrt(size)
      trainFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
    }
    trainFeedBack.close()

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\test\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      (question,user)
    }).sortBy(_._1).collect()

    val testFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\test.txt")
    for ((question,user)<-test){
      val row = questionMap(question)
      val col = userMap(user)
      testFile.println("0 0 1 1 "+row+":1 "+col+":1")
    }
    testFile.close()
    val testFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\test_feed.txt")
    last = questionMap(test(0)._1)
    num = 1
    for (i<-1 until test.length){
      val row = questionMap(test(i)._1)
      if (row == last){
        num += 1
      }else{
        val uNum = q_u.get(last)
        if (uNum == None){
          testFeedBack.println(num+" 0")
        }else{
          testFeedBack.print(num+" ")
          val uArr = uNum.get.toArray.sorted
          val size = uArr.length
          testFeedBack.print(size+" ")
          val value = 1.0/Math.sqrt(size)
          testFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
        }
        last = row
        num = 1
      }
    }
    val uNum2 = q_u.get(last)
    if (uNum2 == None){
      testFeedBack.println(num+" 0")
    }else{
      testFeedBack.print(num+" ")
      val uArr = uNum2.get.toArray.sorted
      val size = uArr.length
      testFeedBack.print(size+" ")
      val value = 1.0/Math.sqrt(size)
      testFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
    }
    testFeedBack.close()
  }
}
