package com

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/22.
  */
object word2Vec {

  def main(args: Array[String]): Unit ={
    val conf = new SparkConf().setAppName("word2Vec").setMaster("local[4]")
    val sc = new SparkContext(conf)
    dataExtract(sc)
    //val model = Word2VecModel.load(sc, "C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\word2Vec")
//    val synonyms = model.findSynonyms("40", 5)
//    for (syn<-synonyms){
//      println(syn._1+" "+syn._2)
//    }
  }
  def dataExtract(sc:SparkContext): Unit ={
    val userInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\user_info.txt")
      .map({x=>
        val info = x.split("\t")
        val user = info(0)
        val tags = info(1).split("/").map(_.toInt)
        val term = info(2).split("/")
        val word = if (info.length == 4){
          info(3).split("/")
        }else{
          new Array[String](0)
        }
        //(user,(tags,term,word))
        (term.toSeq,word.toSeq)
      })

    val questionInfo = sc
      .textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\question_info.txt")
      .map({x=>
        val info = x.split("\t")
        val question = info(0)
        val tags = info(1).split("/").map(_.toInt)
        val term = if (info(2)!=""){
          info(2).split("/")
        }else{
          new Array[String](0)
        }
        val word = if (info(3)!=""){
          info(3).split("/")
        }else{
          new Array[String](0)
        }
        val support = info(4).toInt
        val answer = info(5).toInt
        val goodAnswer = info(6).toInt
        //(question,(tags,term,word,support,answer,goodAnswer))
        (term.toSeq,word.toSeq)
      })
    val user1 = userInfo.map(_._1)
    val user2 = userInfo.map(_._2)
    val q1 = questionInfo.map(_._1)
    val q2 = questionInfo.map(_._2)

    val sentence = user1.union(user2).union(q1).union(q2)
    val word2Vec = new Word2Vec()
    for (vecSize<-Range(60,100,10)){
      for (windowSize<-Range(3,7,1)){
        val model = word2Vec.setMinCount(0).setNumIterations(10).setVectorSize(vecSize).setWindowSize(windowSize).fit(sentence)
        model.save(sc,"C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\word2vec\\word2Vec_10_"+vecSize+"_"+windowSize)
      }
    }


  }
}
