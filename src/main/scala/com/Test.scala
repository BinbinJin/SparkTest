package com

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
    dataTrans(sc,dataName,35)
  }
  def dataTrans(sc:SparkContext,inputName:String,modelNum:Int): Unit ={
    val featureNew = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\featureL1")
      .zipWithIndex()
      .filter(x=>x._1!="0.0" || (x._2>=12875 && x._2<=12886) || (x._2>=44204 && x._2<=44405))
      .map(x=>x._2)
      .zipWithIndex()
      .map(x=>(x._1.toInt,x._2.toInt))
      .collect()
      .toMap
    val featureNewNum = featureNew.size

    val data = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\train\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      val indices = new Array[Int](featureNewNum)
      val value = new Array[Double](featureNewNum)
      for (i<-1 until labelAndFeature.length){
        val indAndVal = labelAndFeature(i).split(":")
        val ind = featureNew.getOrElse(indAndVal(0).toInt-1,-1)
        if (ind != -1) {
          indices(ind) = ind
          value(ind) = indAndVal(1).toDouble
        }
      }
      val feature2 = indices.zip(value).filter(x=>x._2!=0).map(x=>x._1+":"+x._2)
      question+"\t"+user +"\t"+label+" "+feature2.mkString(" ")
    }).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\15_raw+compress_rawQ_rawU_df0_3rel_userPre50+143_non0_L1\\train")

    val test = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\"+inputName+"\\test\\part-*").map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val feature = info(2).split(" ")
      val indices = new Array[Int](featureNewNum)
      val value = new Array[Double](featureNewNum)
      for (i<-0 until feature.length){
        val indAndVal = feature(i).split(":")
        val ind = featureNew.getOrElse(indAndVal(0).toInt-1,-1)
        if (ind != -1) {
          indices(ind) = ind
          value(ind) = indAndVal(1).toDouble
        }
      }
      val feature2 = indices.zip(value).filter(x=>x._2!=0).map(x=>x._1+":"+x._2)
      question+"\t"+user +"\t"+feature2.mkString(" ")
    }).saveAsTextFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\libSVM\\15_raw+compress_rawQ_rawU_df0_3rel_userPre50+143_non0_L1\\test")
print(featureNewNum)
  }
}
