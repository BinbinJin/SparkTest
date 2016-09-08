package com

import java.io.PrintWriter

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
  * Created by JinBinbin on 2016/8/29.
  */
object LR {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("LR").setMaster("local[4]")
    val sc = new SparkContext(conf)

//    val input = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\result.csv").map({x=>
//      val info = x.split(",")
//      val qid = info(0)
//      val uid = info(1)
//      val pre = info(2).toDouble
//      val label = info(3).toInt
//      (qid,uid,pre,label)
//    }).sortBy(x=>(x._1,x._3)).map(x=>x._1+"\t"+x._2+"\t"+x._3+"\t"+x._4).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\haha")

//    val model = LogisticRegressionModel.load(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelSub6")
//    val weights = model.weights.toArray.zipWithIndex.sortBy(x=>x._1)
//    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\weightModel6.txt")
//    out.println(model.numFeatures)
//    for ((weight,index)<-weights){
//      out.write(index+"\t"+weight+"\n")
//    }
//    out.close()
    //dataProcessing(sc)
    train(sc)
    //evaluate(sc)
    //featurePrint(rawFeatureNum,data,testOnline)

    sc.stop()
  }

  def train(sc:SparkContext): Unit ={
    val data = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\15_raw_df7_gbdt_df7_460_4\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      val indices = new Array[Int](labelAndFeature.length-1)
      val value = new Array[Double](labelAndFeature.length-1)
      for (i<-1 until labelAndFeature.length){
        val indAndVal = labelAndFeature(i).split(":")
        indices(i-1) = indAndVal(0).toInt-1
        value(i-1) = indAndVal(1).toDouble
      }
      (label,indices,value)
    })
    val featureNum = data.flatMap(x=>x._2).max()+1
    val training = data.map(x=>LabeledPoint(x._1,Vectors.sparse(featureNum,x._2,x._3))).cache()

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\15_raw_df7_gbdt_df7_460_4\\test\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val feature = info(2).split(" ")
      val indices = new Array[Int](feature.length)
      val value = new Array[Double](feature.length)
      for (i<-0 until feature.length){
        val indAndVal = feature(i).split(":")
        indices(i) = indAndVal(0).toInt-1
        value(i) = indAndVal(1).toDouble
      }
      (question,user,Vectors.sparse(featureNum,indices,value))
    }).cache()
//    val data = parsedData.map(_._3).cache()
//    val testPred = testOnline.cache()
    //val splits = data.randomSplit(Array(0.5,0.5),seed = 11L)
    //val training = splits(0).cache()
    //val test = splits(1).cache()

    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)
    model.save(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelSub14")
    val model2 = LogisticRegressionModel.load(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelSub14").clearThreshold()
    val predictionAndLabel = test
      .map({x=>
        val prediction = model2.predict(x._3).toFloat
        x._1+","+x._2+","+prediction
      })
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\resSub14")
//    val predictionAndLabel = test.map({case LabeledPoint(label,feature)=>
//      val prediction = model.predict(feature)
//      (prediction,label)
//    })
//    predictionAndLabel.map(x=>x._1+"\t"+x._2).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\res0.5_df7Rate")
  }

  def dataProcessing(sc:SparkContext):Unit = {
    val userInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\user_info.txt")
      .map({x=>
        val info = x.split("\t")
        val user = info(0)
        val tags = info(1).split("/").map(_.toInt)
        val term = info(2).split("/").map(_.toInt)
        val word = if (info.length == 4){
          info(3).split("/").map(_.toInt)
        }else{
          new Array[Int](0)
        }
        (user,(tags,term,word))
      })

    val questionInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\question_info.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val tags = info(1).split("/").map(_.toInt)
        val term = if (info(2)!=""){
          info(2).split("/").map(_.toInt)
        }else{
          new Array[Int](0)
        }
        val word = if (info(3)!=""){
          info(3).split("/").map(_.toInt)
        }else{
          new Array[Int](0)
        }
        val support = info(4).toInt
        val answer = info(5).toInt
        val goodAnswer = info(6).toInt
        (question,(tags,term,word,support,answer,goodAnswer))
      })

    val userInfoMap = userInfo.collect().toMap
    val questionInfoMap = questionInfo.collect().toMap

    val invitedInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\label2.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val user = info(1)
        val label = info(2).toInt
        val gbdtFeature = info(3)
        (question,user,label,gbdtFeature)
      })

    /*统计专家回答率，问题被回答率*/
    val userAnswerRate = invitedInfo.map(x=>(x._2,(x._3,1))).reduceByKey((x,y)=>(x._1+y._1,x._2+y._2))/*.map(x=>(x._1,x._2._1*1.0/x._2._2))*/.collect().toMap
    val questionAnsweredRate = invitedInfo.map(x=>(x._1,(x._3,1))).reduceByKey((x,y)=>(x._1+y._1,x._2+y._2))/*.map(x=>(x._1,x._2._1*1.0/x._2._2))*/.collect().toMap

    /*提取原始特征（字、词、标签）*/
    val featureMap = invitedInfo
      .flatMap({x=>
        val question = x._1
        val user = x._2
        val questionInfo = questionInfoMap.getOrElse(question,null)
        val userInfo = userInfoMap.getOrElse(user,null)
        (questionInfo._1++questionInfo._2++questionInfo._3++userInfo._1++userInfo._2++userInfo._3).distinct
      })
      .map(x=>(x,1))
      .reduceByKey(_+_)
      .sortBy(x=>x._2)
      .filter(_._2 > 7)
      .map(_._1)
      .zipWithIndex()
      .map(x=>(x._1,x._2.toInt))
      .collect()
      .toMap
    val rawFeatureNum = featureMap.size

    /*提取由gbdt做出来的特征*/
    val gbdtFeature = invitedInfo.map({x=>
      val s = x._4.replaceAll(",","").split(" ").map(_.toInt)
      s
    })
    val gbdtFeatureMap = gbdtFeature
      .map(x=>x.map(x=>Set(x)))
      .reduce({(x,y)=>
        val combine = for (i<-x.indices) yield {
          x(i)++y(i)
        }
        combine.toArray
      })
      .map(_.toArray.zipWithIndex.toMap)
    var gbdtFeatureNum = 0
    for (map<-gbdtFeatureMap){
      gbdtFeatureNum = gbdtFeatureNum + map.size
    }

//    /*计算所有专家和问题记录（包括有label和没label的数据）的IDF*/
//    val testQAU = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\nolabel.txt").map({x=>
//      val sp = x.split("\t")
//      val info = sp(0).split(",")
//      val question = info(0)
//      val user = info(1)
//      (question,user)
//    })
//    val questionAndUser =invitedInfo
//      .map(x=>(x._1,x._2))
//      .union(testQAU)
//    val recordNum = questionAndUser.count()
//    val DF = questionAndUser
//      .flatMap({x=>
//        val question = x._1
//        val user = x._2
//        val questionInfo = questionInfoMap.getOrElse(question,null)
//        val userInfo = userInfoMap.getOrElse(user,null)
//        val tags = questionInfo._1++questionInfo._2++questionInfo._3++userInfo._1++userInfo._2++userInfo._3
//        tags.distinct
//      })
//      .map(x=>(x,1))
//      .reduceByKey(_+_)
//      .collect()
//    val IDFData = new Array[Double](rawFeatureNum)
//    for ((feature,count)<-DF){
//      val ind = featureMap.getOrElse(feature,-1)
//      if (ind != -1) {
//        IDFData(ind) = Math.log10(recordNum * 1.0 / count)
//      }
//    }

    /*提取没label的专家-问题记录的特征*/
    val testOnline = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\nolabel2.txt").map({x=>
      val sp = x.split("\t")
      val info = sp(0).split(",")
      val question = info(0)
      val user = info(1)
      val gbdtFeatureInd = sp(1).replaceAll(",","").split(" ").map(_.toInt)
      val rawFeature = getRawFeature(question,user,questionInfoMap,questionAnsweredRate,userInfoMap,userAnswerRate,featureMap,rawFeatureNum)
      val gbdtFeature = getGBDTFeature(gbdtFeatureInd,gbdtFeatureMap,gbdtFeatureNum)
      (question,user,Vectors.dense(rawFeature++gbdtFeature))
    })

    /*提取有label的专家-问题记录的特征*/
    val data = invitedInfo.map({x=>
      val question = x._1
      val user = x._2
      val label = x._3
      val gbdtFeatureInd = x._4.replaceAll(",","").split(" ").map(_.toInt)
      val rawFeature = getRawFeature(question,user,questionInfoMap,questionAnsweredRate,userInfoMap,userAnswerRate,featureMap,rawFeatureNum)
      val gbdtFeature = getGBDTFeature(gbdtFeatureInd,gbdtFeatureMap,gbdtFeatureNum)
      (question,user,LabeledPoint(label.toDouble,Vectors.dense(rawFeature++gbdtFeature)))
    })

    data.map(x=>{
      val qid = x._1
      val uid = x._2
      val label = x._3.label
      val feature = x._3.features.toSparse
      val indices = feature.indices.map(x=>x+1)
      val value = feature.values
      val feature2 = indices.zip(value).map(x=>x._1+":"+x._2)
      qid+"\t"+uid+"\t"+label+" "+feature2.mkString(" ")
    }).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\15_raw_df7_gbdt_df7_460_4\\train")
    testOnline.map(x=>{
      val qid = x._1
      val uid = x._2
      val feature = x._3.toSparse
      val indices = feature.indices.map(x=>x+1)
      val value = feature.values
      val feature2 = indices.zip(value).map(x=>x._1+":"+x._2)
      qid+"\t"+uid+"\t"+feature2.mkString(" ")
    }).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\15_raw_df7_gbdt_df7_460_4\\test")
    println(rawFeatureNum+15+" "+ gbdtFeatureNum)
//    (rawFeatureNum+5,gbdtFeatureNum,data,testOnline)
  }

  def getRawFeature(question:String,
                    user:String,
                    questionInfoMap:Map[String,(Array[Int],Array[Int],Array[Int],Int,Int,Int)],
                    questionAnsweredRate:Map[String,(Int,Int)],
                    userInfoMap:Map[String,(Array[Int],Array[Int],Array[Int])],
                    userAnswerRate:Map[String,(Int,Int)],
                    featureMap:Map[Int,Int],
                    rawFeatureNum:Int): Array[Double] ={
    val rawFeature = new Array[Double](rawFeatureNum+15)
    val questionInfo = questionInfoMap.getOrElse(question,null)
    val userInfo = userInfoMap.getOrElse(user,null)
    val tags = questionInfo._1++questionInfo._2++questionInfo._3++userInfo._1++userInfo._2++userInfo._3
    for (tag<-tags){
      val ind = featureMap.getOrElse(tag,-1)
      if (ind != -1) {
        rawFeature(ind) = rawFeature(ind) + 1
      }
    }
    //      for (i<-0 until rawFeatureNum){
    //        rawFeature(i) = rawFeature(i) * 1.0 / tags.length * IDFData(i)
    //      }
    rawFeature(rawFeatureNum) = questionInfo._4
    rawFeature(rawFeatureNum+1) = questionInfo._5
    rawFeature(rawFeatureNum+2) = questionInfo._6
    //rawFeature(rawFeatureNum+3) = questionAnsweredRate.getOrElse(question,0)
    //rawFeature(rawFeatureNum+4) = userAnswerRate.getOrElse(user,0)
    rawFeature(rawFeatureNum+3) = questionInfo._1.length
    rawFeature(rawFeatureNum+4) = questionInfo._2.length
    rawFeature(rawFeatureNum+5) = questionInfo._3.length
    rawFeature(rawFeatureNum+6) = userInfo._1.length
    rawFeature(rawFeatureNum+7) = userInfo._2.length
    rawFeature(rawFeatureNum+8) = userInfo._3.length
    rawFeature(rawFeatureNum+9) = questionAnsweredRate.getOrElse(question,Tuple2(0,0))._1
    rawFeature(rawFeatureNum+10) = questionAnsweredRate.getOrElse(question,Tuple2(0,0))._2
    if (rawFeature(rawFeatureNum+10)!=0){
      rawFeature(rawFeatureNum+11) = rawFeature(rawFeatureNum+9) * 1.0 / rawFeature(rawFeatureNum+10)
    }
    rawFeature(rawFeatureNum+12) = userAnswerRate.getOrElse(user,Tuple2(0,0))._1
    rawFeature(rawFeatureNum+13) = userAnswerRate.getOrElse(user,Tuple2(0,0))._2
    if (rawFeature(rawFeatureNum+13)!=0){
      rawFeature(rawFeatureNum+14) = rawFeature(rawFeatureNum+12) * 1.0 / rawFeature(rawFeatureNum+13)
    }
    rawFeature
  }

  def getGBDTFeature(gbdtFeatureInd:Array[Int],
                     gbdtFeatureMap:Array[Map[Int,Int]],
                     gbdtFeatureNum:Int): Array[Double] ={
    val gbdtFeature = new Array[Double](gbdtFeatureNum)
    var index = 0
    for (i <- gbdtFeatureInd.indices){
      val featureInd = gbdtFeatureInd(i)
      val ind = gbdtFeatureMap(i).get(featureInd).get
      gbdtFeature(index+ind) = 1
      index = index + gbdtFeatureMap(i).size
    }
    gbdtFeature
  }

  def evaluate(sc:SparkContext): Unit ={
    val input = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\res0.5_df8Rate\\part-00000").map{x=>
      val info = x.split("\t")
      val pre = info(0).toDouble
      val label = info(1).toDouble
      (pre,label)
    }
    val total = input.count()
    val cor = input.filter(x=>x._1==x._2).count()
    println(cor+" "+total)
    val precision = input.filter(x=>x._1==1).count()
    val recall = input.filter(x=>x._2==1).count()
    val correct = input.filter(x=>x._2==1&& x._1==x._2).count()
    println(precision+" "+recall+" "+correct)
  }

  def featurePrint(featureNum:Int,
                   data:RDD[(String,String,LabeledPoint)],
                   testOnline:RDD[(String,String,Vector)]): Unit ={
    data.map({x=>
      val res = new Array[String](featureNum)
      val question = x._1
      val user = x._2
      val feature = x._3.features.toArray
      var k = 0
      for (i<-0 until featureNum){
        if (feature(i)>0){
          res(k) = (i+1)+":"+feature(i)
          k = k + 1
        }
      }
      val res2 = res.filter(_!=null)
      question+"\t"+user+"\t"+x._3.label.toInt+" "+res2.mkString(" ")
    }).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data2Vector4")

    testOnline.map({x=>
      val res = new Array[String](featureNum)
      val question = x._1
      val user = x._2
      val feature = x._3.toArray
      var k = 0
      for (i<-0 until featureNum){
        if (feature(i)>0){
          res(k) = (i+1)+":"+feature(i)
          k = k + 1
        }
      }
      val res2 = res.filter(_!=null)
      question+"\t"+user+"\t"+" "+res2.mkString(" ")
    }).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data2VectorTest4")

  }
}
