package com

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by JinBinbin on 2016/8/30.
  */
object Test {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("Test").setMaster("local")
    val sc = new SparkContext(conf)

    val input = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\resdf5\\part-00000")
    input.saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\haha")
  }
}
