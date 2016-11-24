package com

import java.io.PrintWriter

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/10/21.
  */
object gen_feat_RankSVD {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("RankSVD").setMaster("local[3]")
    val sc = new SparkContext(conf)
    val dataName = "qid_uid_0-1"
    //dataTrans_cv(sc,dataName)
    dataTrans_all(sc,dataName)

  }

  /*两种评分*/
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


    val RDDArr = new Array[RDD[String]](8)
    for (i<-0 until 8){
      RDDArr(i) = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\qid_uid_1-4\\part"+i+"\\part-00000").cache()
    }
    for (i<-0 until 8) {
      var trainInput = sc.makeRDD(new Array[String](0))
      for (j <- 0 until 8 if j != i) {
        trainInput = trainInput.union(RDDArr(j))
      }
      val validationInput = RDDArr(i)

      val u_q = trainInput.map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature(0).toDouble
        val qNum = questionMap(question)
        val uNum = userMap(user)
        (label,qNum,uNum)
      }).filter(x=>x._1==3.0 || x._1==4.0).map{x=>(x._3,x._2)}.groupByKey().collect().toMap

      val train = trainInput.map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature(0).toDouble
        val label2 = label match{
          case 1.0 => 1.0
          case 2.0 => 2.0
          case 3.0 => 4.0
          case 4.0 => 3.0
        }
        (label2,question,user)
      }).sortBy(_._3).collect()
      val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\train"+i+".txt")
      for ((label,question,user)<-train){
        val row = questionMap(question)
        val col = userMap(user)
        trainFile.println(label+" 0 1 1 "+col+":1 "+row+":1")
      }
      trainFile.close()
      val trainFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\train_feed"+i+".txt")
      var last = userMap(train(0)._3)
      var num = 1
      for (i<-1 until train.length){
        val row = userMap(train(i)._3)
        if (row == last){
          num += 1
        }else{
          val uNum = u_q.get(last)
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
      val uNum = u_q.get(last)
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

      val validation = validationInput.map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature(0).toDouble
        val label2 = label match{
          case 1.0 => 1.0
          case 2.0 => 2.0
          case 3.0 => 4.0
          case 4.0 => 3.0
        }
        (label2,question,user)
      }).sortBy(_._3).collect()
      val validationFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\validation"+i+".txt")
      for ((label,question,user)<-validation){
        val row = questionMap(question)
        val col = userMap(user)
        validationFile.println(label+" 0 1 1 "+col+":1 "+row+":1")
      }
      validationFile.close()

      val validationFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\cv\\validation_feed"+i+".txt")
      last = userMap(validation(0)._3)
      num = 1
      for (i<-1 until validation.length){
        val row = userMap(validation(i)._3)
        if (row == last){
          num += 1
        }else{
          val uNum = u_q.get(last)
          if (uNum == None){
            validationFeedBack.println(num+" 0")
          }else{
            validationFeedBack.print(num+" ")
            val uArr = uNum.get.toArray.sorted
            val size = uArr.length
            validationFeedBack.print(size+" ")
            val value = 1.0/Math.sqrt(size)
            validationFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
          }
          last = row
          num = 1
        }
      }
      val uNum2 = u_q.get(last)
      if (uNum2 == None){
        validationFeedBack.println(num+" 0")
      }else{
        validationFeedBack.print(num+" ")
        val uArr = uNum2.get.toArray.sorted
        val size = uArr.length
        validationFeedBack.print(size+" ")
        val value = 1.0/Math.sqrt(size)
        validationFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
      }
      validationFeedBack.close()
    }
  }
/*两种评分*/
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
//      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\user")

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
//      .zipWithIndex().saveAsTextFile("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\question")

    val u_q = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      val qNum = questionMap(question)
      val uNum = userMap(user)
      (label,qNum,uNum)
    }).filter(_._1==1.0).map{x=>
      (x._3,x._2)
    }.groupByKey().collect().toMap

    val userSeq = new Array[Double](28763)
    for (i<-0 until 28763){
      userSeq(i) = util.Random.nextDouble()
    }

    val train = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val qNum = questionMap(question).toInt
      val uNum = userMap(user).toInt
      val labelAndFeature = info(2).split(" ")
      val label = if (labelAndFeature(0).toDouble==1.0){1}else{0}
      val order = userSeq(uNum)
      (label,qNum,uNum,order)
    }).sortBy(_._4).map(x=>(x._1,x._2,x._3)).collect()



    val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\train.txt")
    for ((label,qNum,uNum)<-train){
      trainFile.println(label+" 0 1 1 "+uNum+":1 "+qNum+":1")
    }
    trainFile.close()
    val trainFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\train_feed.txt")
    var last = train(0)._3
    var num = 1
    for (i<-1 until train.length){
      val row = train(i)._3
      if (row == last){
        num += 1
      }else{
        val uNum = u_q.get(last)
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
    val uNum = u_q.get(last)
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
      val qNum = questionMap(question).toInt
      val uNum = userMap(user).toInt
      val order = userSeq(uNum)
      (qNum,uNum,order)
    }).sortBy(_._3).map(x=>(x._1,x._2)).collect()

    val testFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\test.txt")
    for ((qNum,uNum)<-test){
      testFile.println("0 0 1 1 "+uNum+":1 "+qNum+":1")
    }
    testFile.close()
    val testFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\test_feed.txt")
    last = test(0)._2
    num = 1
    for (i<-1 until test.length){
      val row = test(i)._2
      if (row == last){
        num += 1
      }else{
        val uNum = u_q.get(last)
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
    val uNum2 = u_q.get(last)
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

    val finalTest = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\test_nolabel.txt").map({x=>
      val info = x.split(",")
      val question = info(0)
      val user = info(1)
      val qNum = questionMap(question).toInt
      val uNum = userMap(user).toInt
      val order = userSeq(uNum)
      (qNum,uNum,order)
    }).sortBy(_._3).map(x=>(x._1,x._2)).collect()

    val finalTestFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\finalTest.txt")
    for ((qNum,uNum)<-finalTest){
      finalTestFile.println("0 0 1 1 "+uNum+":1 "+qNum+":1")
    }
    finalTestFile.close()
    val finalTestFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\fianlTest_feed.txt")
    last = finalTest(0)._2
    num = 1
    for (i<-1 until finalTest.length){
      val row = finalTest(i)._2
      if (row == last){
        num += 1
      }else{
        val uNum = u_q.get(last)
        if (uNum == None){
          finalTestFeedBack.println(num+" 0")
        }else{
          finalTestFeedBack.print(num+" ")
          val uArr = uNum.get.toArray.sorted
          val size = uArr.length
          finalTestFeedBack.print(size+" ")
          val value = 1.0/Math.sqrt(size)
          finalTestFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
        }
        last = row
        num = 1
      }
    }
    val uNum3 = u_q.get(last)
    if (uNum3 == None){
      finalTestFeedBack.println(num+" 0")
    }else{
      finalTestFeedBack.print(num+" ")
      val uArr = uNum3.get.toArray.sorted
      val size = uArr.length
      finalTestFeedBack.print(size+" ")
      val value = 1.0/Math.sqrt(size)
      finalTestFeedBack.println(uArr.map(x=>x+":"+value).mkString(" "))
    }
    finalTestFeedBack.close()

  }
}
