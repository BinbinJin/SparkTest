package com

import java.io.PrintWriter

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/24.
  */
object NDCG {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("modelMerge").setMaster("local")
    val sc = new SparkContext(conf)

//    val fileName = "C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\MF\\cv\\MF_cv_ndcg.txt"
//    val input = sc.textFile(fileName).map{x=>
//      val info = x.split("\t")
//      val random = scala.util.Random.nextDouble()
//      val label = if (info(3).toInt == -1){0} else{1}
//      (info(0),info(1),info(2).toDouble,label)
//    }

    val input1 = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\data\\cv\\validation.txt").map{x=>
      val info = x.split(" ")
      val target = info(0).toInt
      val label = if(target==1){1}else{0}
      val qid = info(4).split(":")(0)
      val uid = info(5).split(":")(0)
      (qid,uid,label)
    }.zipWithIndex().map(_.swap)
    val out = new PrintWriter("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\compare.txt")
    for (i<-30 to 100){
      val input2 = sc.textFile("C:\\Users\\zjcxj\\Documents\\Visual Studio 2015\\Projects\\RankSVD\\RankSVD\\pred\\pred"+i+".txt").zipWithIndex().map(_.swap)
      val input =input1.join(input2).map(x=>(x._2._1._1,x._2._1._2,x._2._2.toDouble,x._2._1._3))
      val ndcg = NDCG(input)
      out.println(i+":"+ndcg);

    }
    out.close()
    //val ndcg = NDCG(input)
    //println(ndcg)
  }

  def NDCG(data:RDD[(String,String,Double,Int)]): Double ={
    val input = data
      .map{x=>
        val qid = x._1
        val uid = x._2
        val pred = x._3
        val label = x._4
        (qid,(uid,pred,label))
      }
      .groupByKey()
      .collect()
    var ndcgSum = 0.0
    for ((_,uSet)<-input){
      val set = uSet.toArray.sortWith((x,y)=>x._2>y._2)
      val (ndcg5,ndcg10) = DCG(set.map(x=>x._3))
      ndcgSum = ndcgSum + ndcg5 * 0.5 + ndcg10 * 0.5
    }
    ndcgSum / input.length
  }

  def DCG(seq:Array[Int]):(Double,Double) ={
    val num5 = Math.min(5,seq.length)
    var dcg5 = 0.0
    for (i<-0 until num5){
      dcg5 = dcg5 + seq(i) * 1.0 / Math.log(i+2)
    }
    val num10 = Math.min(10,seq.length)
    var dcg10 = 0.0
    for (i<-0 until num10){
      dcg10 = dcg10 + seq(i) * 1.0 / Math.log(i+2)
    }
    val seqDec5 = seq.take(num5).sortWith(_>_)
    val seqDec10 = seq.take(num10).sortWith(_>_)
    var idcg5 = 0.0
    for (i<-0 until num5){
      idcg5 = idcg5 + seq(i) * 1.0 / Math.log(i+2)
    }
    var idcg10 = 0.0
    for (i<-0 until num10){
      idcg10 = idcg10 + seq(i) * 1.0 / Math.log(i+2)
    }
    (if (idcg5 > 0) {dcg5 / idcg5}else{0},if (idcg10 > 0) {dcg10 / idcg10} else{0})
  }
}
