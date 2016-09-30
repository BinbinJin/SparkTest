package com

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by JinBinbin on 2016/8/30.
  */
object Test {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("RF").setMaster("local[3]")
    val sc = new SparkContext(conf)
    val dataName = "15_raw+compress_rawQ_rawU_df0_3rel_userPre50+143_non0"
    dataTrans(sc,dataName,44)
  }
  def dataTrans(sc:SparkContext,inputName:String,modelNum:Int): Unit ={
    val input = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\invited_info_train.txt")
    val splits = input.randomSplit(Array(0.9,0.1),11L)
    splits(0).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\train.txt")
    splits(1).repartition(1).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\data\\validation.txt")
  }
}
