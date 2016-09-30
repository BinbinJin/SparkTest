package com

import java.io.PrintWriter

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
  * Created by zjcxj on 2016/9/30.
  */
object gen_feat_cv {
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
    val dataName = "answerRate6_Des6_intersection4_dis163_preferDis33_hot7"
    val embedding = "10_40_3"
    featExtract(sc,dataName,embedding,5)
    //    sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\validate_nolabel.txt").map(x=>x.split(","))
    //      .map(x=>((x(0),x(1)),1)).reduceByKey(_+_).sortByKey().repartition(1).saveAsTextFile("./haha2")
  }
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
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\train.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val label = info(2).toInt
        (question,user,label)
      })
      .cache()

    /*收集user、question信息*/
    val userInfoMap = userInfo.collect().toMap
    val (supportCB,answerCB,goodAnswerCB) = getCutBorder(questionInfo)
    val questionInfoMap = questionInfo.map({x=>
      val support = x._2._4
      val answer = x._2._5
      val goodAnswer = x._2._6
      var k1 = 0
      while (support>supportCB(k1)._2) k1 = k1 + 1
      var k2 = 0
      while (answer>answerCB(k2)._2) k2 = k2 + 1
      var k3 = 0
      while (goodAnswer>goodAnswerCB(k3)._2) k3 = k3 + 1
      (x._1,(x._2._1,x._2._2,x._2._3,k1,k2,k3,Array[Double](support,answer,goodAnswer)))
    }).collect().toMap

    /*统计原始特征数量*/
    val questionFeatureMap = invitedInfo
      .flatMap({x=>
        val question = x._1
        val questionInfo = questionInfoMap.getOrElse(question,null)
        (questionInfo._1++questionInfo._2++questionInfo._3).distinct
      })
      .map(x=>(x,1))
      .reduceByKey(_+_)
      .filter(_._2 > minCount)
      .map(_._1)
      .zipWithIndex()
      .map(x=>(x._1,x._2.toInt))
      .collect()
      .toMap
    val userFeatureMap = invitedInfo
      .flatMap({x=>
        val user = x._2
        val userInfo = userInfoMap.getOrElse(user,null)
        (userInfo._1++userInfo._2++userInfo._3).distinct
      })
      .map(x=>(x,1))
      .reduceByKey(_+_)
      .filter(_._2 > minCount)
      .map(_._1)
      .zipWithIndex()
      .map(x=>(x._1,x._2.toInt))
      .collect()
      .toMap
    val questionFeatureNum = questionFeatureMap.size
    val userFeatureNum = userFeatureMap.size

    /*统计专家回答率，问题被回答率*/
    val userAnswerRate = invitedInfo.map(x=>(x._2,(x._3,1))).reduceByKey((x,y)=>(x._1+y._1,x._2+y._2)).collect().toMap
    val questionAnsweredRate = invitedInfo.map(x=>(x._1,(x._3,1))).reduceByKey((x,y)=>(x._1+y._1,x._2+y._2)).collect().toMap

    /*统计用户回答问题的热门程度*/
    val sagMap = new mutable.HashMap[String,(Array[Double],Array[Double],Array[Double])]()
    val sag = invitedInfo.map({x=>
      val question = x._1
      val user = x._2
      val label = x._3
      val questionInfo = questionInfoMap.getOrElse(question,null)
      (user,(questionInfo._4,questionInfo._5,questionInfo._6,label))
    }).collect()
    for ((user,(supprot,answer,goodAnswer,label))<-sag){
      val info = sagMap.getOrElse(user,Tuple3(new Array[Double](10),new Array[Double](10),new Array[Double](10)))
      if (label == 1) {
        info._1(supprot) = info._1(supprot) + 1
        info._2(answer) = info._2(answer) + 1
        info._3(goodAnswer) = info._3(answer) + 1
      }
      sagMap.put(user,info)
    }

    /*统计专家回答不同问题的标签*/
    val userQLabelMap = new mutable.HashMap[String,(Array[Double],Array[Double],Array[Double],Array[Double])]()
    val userPrefer = invitedInfo.map({x=>
      val question = x._1
      val user = x._2
      val label = x._3
      val questionInfo = questionInfoMap.getOrElse(question,null)
      (user,(questionInfo._1(0),label))
    }).collect()
    for ((user,(qlabel,label))<-userPrefer){
      val info = userQLabelMap.getOrElse(user,Tuple4(new Array[Double](20),new Array[Double](20),new Array[Double](20),new Array[Double](20)))
      if (label == 1) {
        info._1(qlabel) = info._1(qlabel) + 1
      }
      info._2(qlabel) = info._2(qlabel) + 1
      info._3(qlabel) = info._1(qlabel) * 1.0 / info._2(qlabel)
      val userAnswerNum = userAnswerRate(user)._1
      if (userAnswerNum!=0) {
        info._4(qlabel) = info._1(qlabel) * 1.0 / userAnswerNum
      }
      userQLabelMap.put(user,info)
    }

    /*统计问题被回答的专家的标签*/
    val questionULbaleMap = new mutable.HashMap[String,(Array[Double],Array[Double],Array[Double],Array[Double])]()
    val questionPrefer = invitedInfo.map{x=>
      val question = x._1
      val user = x._2
      val label = x._3
      val userInfo = userInfoMap.getOrElse(user,null)
      (question,userInfo._1,label)
    }.collect()
    for ((question,userid,label)<-questionPrefer){
      val info = questionULbaleMap.getOrElse(question,(new Array[Double](143),new Array[Double](143),new Array[Double](143),new Array[Double](143)))
      for (uLabel<-userid){
        if (label == 1) {
          info._1(uLabel) = info._1(uLabel) + 1
        }
        info._2(uLabel) = info._2(uLabel) + 1
        info._3(uLabel) = info._1(uLabel) * 1.0 / info._2(uLabel)
        val questionAnsweredNum = questionAnsweredRate(question)._1
        if (questionAnsweredNum!=0){
          info._4(uLabel) = info._1(uLabel) * 1.0 / questionAnsweredNum
        }
      }
      questionULbaleMap.put(question,info)
    }

    /*使用embedding*/
    //    val model = Word2VecModel.load(sc, "C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\word2vec\\word2Vec_"+embeddingName)
    //    val word2vecMap = model.getVectors
    //    val vecSize = word2vecMap("0").length

    /*使用qid和uid*/
    val qid = questionInfo.map(x=>x._1)
    val uid = userInfo.map(x=>x._1)
    val uidFeatMap = uid.zipWithIndex().collect().toMap
    val qidFeatMap = qid.zipWithIndex().collect().toMap
    val idFeatMap = qid.union(uid).zipWithIndex().collect().toMap

    /*对自己设计的特征进行归一化*/
    //    val norm = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\统计\\dis\\min_max.txt").map{x=>
    //      val info = x.split(" ")
    //      val ind = info(0).toInt
    //      val min = info(1).toDouble
    //      val max = info(2).toDouble
    //      (ind,(min,max))
    //    }.collect()

    /*对于qid，uid对，将对应回答的id全部加进去*/
    val qid_uidArr = invitedInfo.filter(x=> x._3==1).map{x=>(x._1,x._2)}.groupByKey().collect().toMap
    val uid_qidArr = invitedInfo.filter(x=> x._3==1).map(x=>(x._2,x._1)).groupByKey().collect().toMap

    val sagClassTotal = questionInfo.map { x =>
      val label = x._2._1(0)
      val support = x._2._4
      val answer = x._2._5
      val goodAnswer = x._2._6
      (label, (support, answer, goodAnswer))
    }.reduceByKey((x,y)=>(x._1+y._1,x._2+y._2,x._3+y._3)).collect().toMap

    val train = invitedInfo.map({x=>
      val question = x._1
      val user = x._2
      val label = x._3

//      val idFeat = genIdFeat(question,user,idFeatMap)
//
//      val qid_uidarr = genCorFeat(question,uidFeatMap,qid_uidArr)
//      val uid_qidarr = genCorFeat(user,qidFeatMap,uid_qidArr)
      //      val qInd = qidFeatMap(question)
      //      val uInd = uidFeatMap(user)
      //      qid_uidarr(uInd.toInt) = 1
      //      uid_qidarr(qInd.toInt) = 1


//      val rawQuestionFeat = genRawFeature(question,null,questionInfoMap,userInfoMap,questionFeatureMap,questionFeatureNum)
//      val rawUserFeat = genRawFeature(null,user,questionInfoMap,userInfoMap,userFeatureMap,userFeatureNum)

      val qDes = (questionInfoMap(question)._1,questionInfoMap(question)._2,questionInfoMap(question)._3)
      val uDes = (userInfoMap(user)._1,userInfoMap(user)._2,userInfoMap(user)._3)
      //      val word2vecQFeat = genWord2VecFeat(qDes,word2vecMap,vecSize)
      //      val word2vecUFeat = genWord2VecFeat(uDes,word2vecMap,vecSize)
      //
      //      val labelFeat = genLabelFeat(qDes._1,uDes._1)

      val qAnswerRate = genAnswerRateFeat(question,questionAnsweredRate)
      val uAnswerRate = genAnswerRateFeat(user,userAnswerRate)

      val qDesNum = genDescribeNum(question,qDes)
      val uDesNum = genDescribeNum(user,uDes)

      val intersectionNum = genIntersectionNum(qDes,uDes)

      val uAnswerDis = userQLabelMap.getOrElse(user,(new Array[Double](20),new Array[Double](20),new Array[Double](20),Array[Double](20)))
      val qAnswerDis = questionULbaleMap.getOrElse(question,(new Array[Double](143),new Array[Double](143),new Array[Double](143),new Array[Double](143)))
      val dis = uAnswerDis._1/*++uAnswerDis._2++uAnswerDis._3++uAnswerDis._4*/++qAnswerDis._1/*++qAnswerDis._2++qAnswerDis._3++qAnswerDis._4*/

      val uPreferDis = genUPrefer(question,user,questionInfoMap,sagMap)

      val hot = genHotFeat(question,questionInfoMap,sagClassTotal)

      val featAll = qAnswerRate++uAnswerRate++qDesNum++uDesNum++intersectionNum++dis++uPreferDis++hot
      //      for ((ind,(min,max))<- norm){
      //        featAll(ind) = (featAll(ind)-min) / (max-min)
      //      }

      (question,user,LabeledPoint(label.toDouble,Vectors.dense(featAll)))
    })

    val testOffline = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\validation.txt").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val label = info(2).toInt

//      val idFeat = genIdFeat(question,user,idFeatMap)
//
//      val qid_uidarr = genCorFeat(question,uidFeatMap,qid_uidArr)
//      val uid_qidarr = genCorFeat(user,qidFeatMap,uid_qidArr)
      //      val qInd = qidFeatMap(question)
      //      val uInd = uidFeatMap(user)
      //      qid_uidarr(uInd.toInt) = 1
      //      uid_qidarr(qInd.toInt) = 1

//      val rawQuestionFeat = genRawFeature(question,null,questionInfoMap,userInfoMap,questionFeatureMap,questionFeatureNum)
//      val rawUserFeat = genRawFeature(null,user,questionInfoMap,userInfoMap,userFeatureMap,userFeatureNum)

      val qDes = (questionInfoMap(question)._1,questionInfoMap(question)._2,questionInfoMap(question)._3)
      val uDes = (userInfoMap(user)._1,userInfoMap(user)._2,userInfoMap(user)._3)
      //      val word2vecQFeat = genWord2VecFeat(qDes,word2vecMap,vecSize)
      //      val word2vecUFeat = genWord2VecFeat(uDes,word2vecMap,vecSize)

      //      val labelFeat = genLabelFeat(qDes._1,uDes._1)

      val qAnswerRate = genAnswerRateFeat(question,questionAnsweredRate)
      val uAnswerRate = genAnswerRateFeat(user,userAnswerRate)

      val qDesNum = genDescribeNum(question,qDes)
      val uDesNum = genDescribeNum(user,uDes)

      val intersectionNum = genIntersectionNum(qDes,uDes)

      val uAnswerDis = userQLabelMap.getOrElse(user,(new Array[Double](20),new Array[Double](20),new Array[Double](20),Array[Double](20)))
      val qAnswerDis = questionULbaleMap.getOrElse(question,(new Array[Double](143),new Array[Double](143),new Array[Double](143),new Array[Double](143)))
      val dis = uAnswerDis._1/*++uAnswerDis._2++uAnswerDis._3++uAnswerDis._4*/++qAnswerDis._1/*++qAnswerDis._2++qAnswerDis._3++qAnswerDis._4*/

      val uPreferDis = genUPrefer(question,user,questionInfoMap,sagMap)

      val hot = genHotFeat(question,questionInfoMap,sagClassTotal)

      val featAll = qAnswerRate++uAnswerRate++qDesNum++uDesNum++intersectionNum++dis++uPreferDis++hot
      //      for ((ind,(min,max))<- norm){
      //        featAll(ind) = (featAll(ind)-min) / (max-min)
      //      }
      (question,user,Vectors.dense(featAll),label)
    })

    //staDis(train)
//    pearson(train)

    train.map(x=>{
      val qid = x._1
      val uid = x._2
      val label = x._3.label
      val feature = x._3.features.toSparse
      val indices = feature.indices.map(x=>x+1)
      val value = feature.values
      val feature2 = indices.zip(value).map(x=>x._1+":"+x._2)
      qid+"\t"+uid+"\t"+label+" "+feature2.mkString(" ")
    }).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+outName+"\\train")
    testOffline.map(x=>{
      val qid = x._1
      val uid = x._2
      val feature = x._3.toSparse
      val label = x._4
      val indices = feature.indices.map(x=>x+1)
      val value = feature.values
      val feature2 = indices.zip(value).map(x=>x._1+":"+x._2)
      qid+"\t"+uid+"\t"+label+" "+feature2.mkString(" ")
    }).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\"+outName+"\\test")
    println(train.first()._3.features.size)
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
