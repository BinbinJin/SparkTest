package com

import java.io.PrintWriter

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/9/23.
  */
object modelMerge {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("modelMerge").setMaster("local[3]")
    val sc = new SparkContext(conf)

    //intuition(sc)
    //test(sc)
    //normalize(sc)
    finalTest(sc)
//    sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\res.txt").map{x=>
//      val info = x.split(" ")
//      val k = info.take(6)
//      val ndcg = info(6).toDouble
//      (k,ndcg)
//    }.sortBy(_._2).map(x=>x._1.mkString(" ")+x._2).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\res_sorted.txt")
  }

  def test(sc:SparkContext): Unit ={
    val input1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\FMmodel\\train.txt").cache()
    val input2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\FMCRRmodel\\train.txt").cache()
    val input3 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\PairwiseFM\\train.txt").cache()
    val input4 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\MF\\train.txt").cache()
    val input5 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD\\train.txt").cache()
    val input6 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD++\\train.txt").cache()

    val qid_uid_label = input1.map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val label = info(2).toDouble.toInt
      ((qid,uid),label)
    }.cache()
    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\res.txt")
    for (k1<-1 until 4;k2<-1 until 4;k3<-1 until 4;k4<-1 until 4;k5<-1 until 4;k6<-1 until 4 ){
      val per = Array(2,0,0,1,1,1)
      val score1 = input1.map{ x=>
        val info = x.split(",")
        val qid = info(0)
        val uid = info(1)
        val score = info(3).toDouble
        ((qid,uid),score * per(0))
      }.distinct()
      val score2 = input2.map{ x=>
        val info = x.split(",")
        val qid = info(0)
        val uid = info(1)
        val score = info(3).toDouble
        ((qid,uid),score * per(1))
      }.distinct()
      val score3 = input3.map{ x=>
        val info = x.split(",")
        val qid = info(0)
        val uid = info(1)
        val score = info(3).toDouble
        ((qid,uid),score * per(2))
      }.distinct()
      val score4 = input4.map{ x=>
        val info = x.split(",")
        val qid = info(0)
        val uid = info(1)
        val score = info(2).toDouble
        ((qid,uid),score * per(3))
      }.distinct()
      val score5 = input5.map{ x=>
        val info = x.split(",")
        val qid = info(0)
        val uid = info(1)
        val score = info(2).toDouble
        ((qid,uid),score * per(4))
      }.distinct()
      val score6 = input6.map{ x=>
        val info = x.split(",")
        val qid = info(0)
        val uid = info(1)
        val score = info(2).toDouble
        ((qid,uid),score * per(5))
      }.distinct()
      val score = score1.union(score2).union(score3).union(score4).union(score5).union(score6).reduceByKey(_+_)
      val data = qid_uid_label.join(score).map(x=>(x._1._1,x._1._2,x._2._2,x._2._1))
      val ndcg = NDCG.NDCG(data)
//    println(ndcg)
      out.println(k1+" "+k2+" "+k3+" "+k4+" "+k5+" "+k6+" "+ndcg)
    }
    out.close()

  }

  def intuition(sc:SparkContext): Unit ={
    val per = Array(0,18,1,12,6,18)
    val input1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FM.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(0))
    }
    val input2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FMCRR.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(1))
    }
    val input3 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FMPairwise.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(2))
    }
    val input4 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\MF.csv").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(3))
    }
    val input5 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\SVD.csv").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(4))
    }
    val input6 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\SVD++.csv").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(5))
    }
    input1.union(input2).union(input3).union(input4).union(input5).union(input6)
      .reduceByKey(_+_)
      .map(x=>x._1._1+","+x._1._2+","+x._2)
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FMCRR_FMPairwise_MF_SVD_SVD++_18-1-12-6-18")
  }

  def normalize(sc:SparkContext): Unit ={
    val per = Array(0,0,1,0,0,2)
    val fileName = Array("FM.txt","FMCRR.txt","FMPairwise.txt","MF.csv","SVD.csv","SVD++.csv")
    val pred = new Array[RDD[((String,String),Double)]](fileName.length)
    for (i<-fileName.indices) {
      pred(i) = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\"+fileName(i)).map{ x=>
        val info = x.split(",")
        val qid = info(0)
        val uid = info(1)
        val score = info(2).toDouble
        ((qid,uid),score)
      }.cache()
    }
    val pred_norm = new Array[RDD[((String,String),Double)]](fileName.length)
    for (i<-fileName.indices) {
      pred_norm(i) = liner(pred(i))
    }
    for (i<-fileName.indices) {
      pred_norm(i) = pred_norm(i).mapValues(x=> x * per(i))
    }
    var res = sc.makeRDD(new Array[((String,String),Double)](0))
    for (i<-fileName.indices) {
      res = res.union(pred_norm(i))
    }
    res.reduceByKey(_+_).map(x=>x._1._1+","+x._1._2+","+x._2).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FM_FMCRR_FMPairwise_MF_SVD_SVD++_norm_0-0-1-0-0-2")
  }

  def liner(data:RDD[((String,String),Double)]):RDD[((String,String),Double)] = {
    val min = data.map(_._2).min()
    val max = data.map(_._2).max()
    data.mapValues(x=>(x-min)/(max-min))
  }

  def finalTest(sc:SparkContext): Unit ={
    val per = Array(0,18,1,12,6,18)
    val input1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\FM\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(0))
    }
    val input2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\FMCRR\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(1))
    }
    val input3 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\PairwiseFM\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(2))
    }
    val input4 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\MF\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(3))
    }
    val input5 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\SVD\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(4))
    }
    val input6 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\SVD++\\test.txt").map{ x=>
      val info = x.split(",")
      val qid = info(0)
      val uid = info(1)
      val score = info(2).toDouble
      ((qid,uid),score * per(5))
    }
    input1.union(input2).union(input3).union(input4).union(input5).union(input6)
      .reduceByKey(_+_)
      .map(x=>x._1._1+","+x._1._2+","+x._2)
      .repartition(1)
      .saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\finalTest\\0-18-1-12-6-18-final")
  }
}
