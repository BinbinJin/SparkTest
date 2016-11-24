package com

import java.io.PrintWriter

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/11/3.
  */
object correlation_coefficient {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("cc").setMaster("local")
    val sc = new SparkContext(conf)
//    val folderName = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\fileName.txt").collect()
//    val out = new PrintWriter("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\res.txt")
//    for (folder1 <- folderName;folder2 <-folderName if folder1!=folder2){
//      val data1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\"+folder1+"\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
//      val data2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\finalTestResult\\"+folder2+"\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
//      val cc = correlation_coefficient(data1,data2)
//      out.println(folder1+"\t"+folder2+"\t"+cc)
//    }
//    out.close()
//    val data1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FMPairwise_513_1-6.csv").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
//    val data2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\modelMerge\\FM.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
//    val data3 = sc.textFile("E:\\SVD++\\merge\\SVD18_3\\score\\part-00000.csv").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
//    val cc1 = correlation_coefficient(data1,data2)
//    val cc2 = correlation_coefficient(data1,data3)
//    val cc3 = correlation_coefficient(data2,data3)
//    println(cc1)
    val data1 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\FMmodel\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
    val data2 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\FMCRRmodel\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
    val data3 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\PairwiseFM\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
    val data4 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\MF\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
    val data5 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
    val data6 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\SVD++\\test.txt").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
    val data7 = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\stacking\\LambdaMART.csv").map(x=>x.split(",")).map(x=>(x(0),x(1),x(2).toDouble))
    val cc1 = correlation_coefficient(data1,data7)
    val cc2 = correlation_coefficient(data2,data7)
    val cc3 = correlation_coefficient(data3,data7)
    val cc4 = correlation_coefficient(data4,data7)
    val cc5 = correlation_coefficient(data5,data7)
    val cc6 = correlation_coefficient(data6,data7)
    println(cc1+" "+cc2+" "+cc3+" "+cc4+" "+cc5+" "+cc6)
//    println(cc4+" "+cc5)
//    println(cc6)
  }

  def correlation_coefficient(data1:RDD[(String,String,Double)],data2:RDD[(String,String,Double)]): Double ={
    val data = data1.map(x=>((x._1,x._2),x._3)).join(data2.map(x=>((x._1,x._2),x._3))).map(x=>(x._2._1,x._2._2)).cache()
    val count = data.count()
    val avgX = data.map(_._1).reduce(_+_) * 1.0 / count
    val avgY = data.map(_._2).reduce(_+_) * 1.0 / count
    val sum1 = data.map{x=>
      val a = x._1
      val b = x._2
      (a-avgX)*(b-avgY)
    }.reduce(_+_)
    val sum2 = data.map{x=>
      val a = x._1
      (a-avgX)*(a-avgX)
    }.reduce(_+_)
    val sum3 = data.map{x=>
      val a = x._2
      (a-avgY)*(a-avgY)
    }.reduce(_+_)
    sum1 / (Math.sqrt(sum2) * Math.sqrt(sum3))
  }
}
