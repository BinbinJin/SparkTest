package com

import java.io.PrintWriter

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
  * Created by JinBinbin on 2016/8/30.
  */
object Test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("genFeatCV").setMaster("local[4]")
    val sc = new SparkContext(conf)
    //    for (vecSize<-Range(60,100,10)){
    //      for (windowSize<-Range(3,7,1)){
    //        val dataName = "word2vec_"+vecSize+"_"+windowSize
    //        val embedding = "10_"+vecSize+"_"+windowSize
    //        featExtract(sc,dataName,embedding,5)
    //      }
    //    }
    val dataName = "qid_uid_rate"
    val embedding = "10_40_3"
    //featExtract(sc,dataName,embedding,5)
    dataTrans_all(sc,dataName)
    //    sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\validate_nolabel.txt").map(x=>x.split(","))
    //      .map(x=>((x(0),x(1)),1)).reduceByKey(_+_).sortByKey().repartition(1).saveAsTextFile("./haha2")
  }

  /*all4种评分*/
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

    val train = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\train\\part-*")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val labelAndFeature = info(2).split(" ")
        val label = labelAndFeature(0).toDouble
        val target = if (label == 1.0){1} else {0}
        ((question,user),(target,1))
      })
      .reduceByKey((x,y)=>(x._1+y._1,x._2+y._2))
      .map{x=>
        val qid = x._1._1
        val uid = x._1._2
        val target = (x._2._1,x._2._2) match {
          case (0,1) => 2
          case (0,2) => 1
          case (1,2) => 3
          case (1,1) => 4
        }
        (target,qid,uid)
      }
      .sortBy(_._3).collect()

    val trainFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\train.txt")
    for ((label,question,user)<-train){
      val row = questionMap(question)
      val col = userMap(user)
      trainFile.println(label+" 0 1 1 "+col+":1 "+row+":1")
    }
    trainFile.close()
    val trainFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\train_feed.txt")
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

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\qid_uid_bin\\test\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      (question,user)
    }).sortBy(_._2).collect()

    val testFile = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\test.txt")
    for ((question,user)<-test){
      val row = questionMap(question)
      val col = userMap(user)
      testFile.println("0 0 1 1 "+col+":1 "+row+":1")
    }
    testFile.close()
    val testFeedBack = new PrintWriter("C:\\Users\\zjcxj\\IdeaProjects\\svdfeature-1.2.2\\data_RankSVD\\all\\test_feed.txt")
    last = userMap(test(0)._2)
    num = 1
    for (i<-1 until test.length){
      val row = userMap(test(i)._2)
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
  }

  /*cv 四种评分*/
  def featExtract(sc:SparkContext,outName:String,embeddingName:String,minCount:Int): Unit ={
    val userInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\user_info.txt")
      .map({x=>
        val info = x.split("\t")
        val user = info(0)
        val tags = info(1).split("/").map(_.toInt)
        val word = info(2).split("/").map(_.toInt)
        val char = info(3).split("/").map(_.toInt)
        (user,(tags,word,char))
      }).cache()

    val questionInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\question_info.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val tags = info(1).split("/").map(_.toInt)
        val word = info(2).split("/").map(_.toInt)
        val char = info(3).split("/").map(_.toInt)
        val support = info(4).toInt
        val answer = info(5).toInt
        val goodAnswer = info(6).toInt
        (question,(tags,word,char,support,answer,goodAnswer))
      }).cache()

    val invitedInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\invited_info_train.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val label = info(2).toInt
        ((question,user),(label,1))
      })
      .reduceByKey((x,y)=>(x._1+y._1,x._2+y._2))
      .map{x=>
        val qid = x._1._1
        val uid = x._1._2
        val target = (x._2._1,x._2._2) match {
          case (0,1) => 2
          case (0,2) => 1
          case (1,2) => 3
          case (1,1) => 4
        }
        (qid,uid,target)
      }
      .cache()

    /*使用qid和uid*/
    val qid = questionInfo.map(x=>x._1)
    val uid = userInfo.map(x=>x._1)
    val idFeatMap = qid.union(uid).zipWithIndex().collect().toMap

    val train_all = invitedInfo.map({x=>
      val question = x._1
      val user = x._2
      val label = x._3

      val idFeat = genIdFeat(question,user,idFeatMap)

      val featAll = idFeat

      (question,user,LabeledPoint(label.toDouble,Vectors.dense(featAll)))
    })

    val splits = train_all.randomSplit(Array(0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125),seed = 11L)
    for (i<-0 until 8){
      val part = splits(i)
      part.map(x=>{
        val qid = x._1
        val uid = x._2
        val label = x._3.label
        val feature = x._3.features.toSparse
        val indices = feature.indices.map(x=>x+1)
        val value = feature.values
        val feature2 = indices.zip(value).map(x=>x._1+":"+x._2)
        qid+"\t"+uid+"\t"+label+" "+feature2.mkString(" ")
      }).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+outName+"\\part"+i)
    }
    println(train_all.count)
  }

  def genLabelFeat(qLabel:Array[Int],uLabel:Array[Int]): Array[Double] ={
    val q = new Array[Double](20)
    val u = new Array[Double](143)
    for (label<-qLabel){
      q(label) = 1
    }
    for (label<-uLabel){
      u(label) = 1
    }
    q++u
  }

  def genUPrefer(question:String,
                 user:String,
                 questionInfoMap:Map[String,(Array[Int],Array[Int],Array[Int],Int,Int,Int,Array[Double])],
                 sagMap:mutable.HashMap[String,(Array[Double],Array[Double],Array[Double])]): Array[Double] ={
    val qInfo = questionInfoMap(question)
    val feat = new Array[Double](3)
    feat(0) = qInfo._4
    feat(1) = qInfo._5
    feat(2) = qInfo._6
    val featT = sagMap.getOrElse(user,(new Array[Double](10),new Array[Double](10),new Array[Double](10)))
    feat++featT._1++featT._2++featT._3
  }

  def genCorFeat(question:String,uidFeatMap:Map[String,Long],qid_uid:Map[String,Iterable[String]]): Array[Double] ={
    val size = uidFeatMap.size
    val feat = new Array[Double](size)
    val uidArr = qid_uid.getOrElse(question,null)
    if (uidArr!=null) {
      for (uid <- uidArr) {
        val ind = uidFeatMap(uid).toInt
        feat(ind) = 1
      }
    }
    feat
  }

  def genIdFeat(question:String,user:String,idFeatMap:Map[String,Long]): Array[Double] ={
    val featNum = idFeatMap.size
    val feat = new Array[Double](featNum)
    val qidInd = idFeatMap(question)
    val uidInd = idFeatMap(user)
    feat(qidInd.toInt) = 1
    feat(uidInd.toInt) = 1
    feat
  }

  def genHotFeat(question:String,
                 questionInfoMap:Map[String,(Array[Int],Array[Int],Array[Int],Int,Int,Int,Array[Double])],
                 sagClassTotal:Map[Int,(Int,Int,Int)]): Array[Double] ={
    val feat = new Array[Double](7)
    val sag = questionInfoMap(question)._7
    val label = questionInfoMap(question)._1(0)
    val sagTotal = sagClassTotal(label)
    feat(0) = sag(0)
    feat(1) = sag(1)
    feat(2) = sag(2)
    feat(3) = if (feat(1)!=0) {feat(2) * 1.0 / feat(1)} else {0}
    feat(4) = feat(0) * 1.0 / sagTotal._1
    feat(5) = feat(1) * 1.0 / sagTotal._2
    feat(6) = feat(2) * 1.0 / sagTotal._3
    feat
  }

  def genIntersectionNum(qDes:(Array[Int],Array[Int],Array[Int]),uDes:(Array[Int],Array[Int],Array[Int])): Array[Double] ={
    val feat = new Array[Double](4)
    feat(0) = qDes._1.intersect(uDes._1).length
    feat(1) = qDes._2.intersect(uDes._2).length
    feat(2) = qDes._3.intersect(uDes._3).length
    feat(3) = feat(0) + feat(1) + feat(2)
    feat
  }

  def genDescribeNum(id:String,des:(Array[Int],Array[Int],Array[Int])): Array[Double] ={
    val feat = new Array[Double](3)
    feat(0) = des._1.length
    feat(1) = des._2.length
    feat(2) = des._3.length
    feat
  }

  def genAnswerRateFeat(id:String,rateMap:Map[String,(Int,Int)]): Array[Double] ={
    val feat = new Array[Double](3)
    feat(0) = rateMap.getOrElse(id,Tuple2(0,0))._1
    feat(1) = rateMap.getOrElse(id,Tuple2(0,0))._2
    feat(2) = if (feat(1)!=0) {feat(0) * 1.0 / feat(1)} else {0}
    feat
  }

  def genWord2VecFeat(des:(Array[Int],Array[Int],Array[Int]),word2vecMap:Map[String,Array[Float]],vecSize:Int): Array[Double] ={

    val word2Vec = new Array[Double](vecSize)
    val tags = des._1++des._2++des._3
    for (tag<-tags){
      val vec = word2vecMap(tag.toString)
      for (i<-0 until vecSize){
        word2Vec(i) = word2Vec(i) + vec(i)
      }
    }
    word2Vec
  }

  def genRawFeature(question:String,
                    user:String,
                    questionInfoMap:Map[String,(Array[Int],Array[Int],Array[Int],Int,Int,Int,Array[Double])],
                    userInfoMap:Map[String,(Array[Int],Array[Int],Array[Int])],
                    featureMap:Map[Int,Int],
                    rawFeatureNum:Int): Array[Double] ={
    val rawFeature = new Array[Double](rawFeatureNum)
    val questionInfo = questionInfoMap.getOrElse(question,null)
    val userInfo = userInfoMap.getOrElse(user,null)
    val tags = if (user==null){
      questionInfo._1++questionInfo._2++questionInfo._3
    }else{
      userInfo._1++userInfo._2++userInfo._3
    }
    for (tag<-tags){
      val ind = featureMap.getOrElse(tag,-1)
      if (ind != -1) {
        rawFeature(ind) = rawFeature(ind) + 1
      }
    }
    if (user==null){
      rawFeature
    }else{
      rawFeature
    }
  }


  def getCutBorder(questionInfo:RDD[(String,(Array[Int],Array[Int],Array[Int],Int,Int,Int))]): (Array[(Int,Int)],Array[(Int,Int)],Array[(Int,Int)]) ={
    val support = borderAnalyse(questionInfo,4)
    val answer = borderAnalyse(questionInfo,5)
    val goodAnswer = borderAnalyse(questionInfo,6)
    (support,answer,goodAnswer)
  }

  def borderAnalyse(questionInfo:RDD[(String,(Array[Int],Array[Int],Array[Int],Int,Int,Int))],no:Int):Array[(Int,Int)] = {
    val splitsNum = 10
    val cutInfo = new Array[(Int,Int)](splitsNum)
    val questionNum = questionInfo.count()
    val attAndCount = no match {
      case 4 => questionInfo.map(x => (x._2._4, 1)).reduceByKey(_ + _).sortByKey().collect()
      case 5 => questionInfo.map(x => (x._2._5, 1)).reduceByKey(_ + _).sortByKey().collect()
      case 6 => questionInfo.map(x => (x._2._6, 1)).reduceByKey(_ + _).sortByKey().collect()
    }
    var k = 0
    var sum = 0
    var min = 0
    var max = 0
    for (i<-0 until attAndCount.length){
      val support = attAndCount(i)._1
      val count = attAndCount(i)._2
      if (sum < questionNum/splitsNum){
        if (sum == 0){
          min = support
        }
        sum = sum + count
      }else{
        max = support - 1
        cutInfo(k) = (min,max)
        min = support
        sum = count
        k = k + 1
      }
    }
    max = attAndCount(attAndCount.length-1)._1
    cutInfo(k) = (min,max)
    cutInfo
  }

  def staDis(data:RDD[(String,String,LabeledPoint)]): Unit ={
    val size = data.first()._3.features.size
    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\统计\\dis\\min_max.txt")
    for (i<-0 until size){
      val value = data.map(x=>x._3.features(i))
      val max = value.max()
      val min = value.min()
      out.println((i+67060)+" "+min+" "+max)
    }
    out.close()
  }

  def pearson(data:RDD[(String,String,LabeledPoint)]): Unit ={
    val data2 = data.cache()
    val size = data2.first()._3.features.size
    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\统计\\pearson.txt")
    for (i<-0 until size){
      val valAndTar = data.map(x=>(x._3.features(i),x._3.label)).collect()
      val valAvg = valAndTar.map(x=>x._1).sum / valAndTar.length
      val TarAvg = valAndTar.map(x=>x._2).sum / valAndTar.length
      var sum1 = 0.0
      var sum2 = 0.0
      var sum3 = 0.0
      for (j<-valAndTar.indices){
        sum1 = sum1 + (valAndTar(j)._1-valAvg)*(valAndTar(j)._2-TarAvg)
        sum2 = sum2 + (valAndTar(j)._1-valAvg)*(valAndTar(j)._1-valAvg)
        sum3 = sum3 + (valAndTar(j)._2-TarAvg)*(valAndTar(j)._2-TarAvg)
      }
      val res = sum1/(Math.sqrt(sum2)*Math.sqrt(sum3))
      out.println(i+" "+ res)
    }
    out.close()
  }
}
