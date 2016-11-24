package com

import java.io.PrintWriter

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zjcxj on 2016/11/9.
  */
object SVD {
  def main(args:Array[String]): Unit ={
    val conf = new SparkConf().setAppName("SVD").setMaster("local[3]")
    val sc = new SparkContext(conf)

    decompotion(sc)
  }

  def decompotion(sc:SparkContext): Unit ={
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

    val RDDArr = new Array[RDD[String]](8)
    for (i<-0 until 8){
      RDDArr(i) = sc.textFile("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\cv\\libSVM\\qid_uid_1-4\\part"+i+"\\part-00000").cache()
    }
    val SVDArr = new Array[Array[Double]](8095)
    for (i<-0 until 8095){
      SVDArr(i) = new Array[Double](28763)
    }
    var trainInput = sc.makeRDD(new Array[String](0))
    for (i<-0 until 8) {
      trainInput = trainInput.union(RDDArr(i))
    }
    val train = trainInput.map({x=>
      val info = x.split("\t")
      val question = info(0)
      val user = info(1)
      val qNum = questionMap(question).toInt
      val uNum = userMap(user).toInt
      val labelAndFeature = info(2).split(" ")
      val label = labelAndFeature(0).toDouble
      (label,qNum,uNum)
    }).collect()
    for ((label,qNum,uNum)<-train){
      SVDArr(qNum)(uNum) = label
    }
    val SVDData = new Array[Vector](8095)
    for (i<-0 until 8095){
      SVDData(i) = Vectors.dense(SVDArr(i))
    }

    val dataRDD = sc.parallelize(SVDData, 2)

    val mat: RowMatrix = new RowMatrix(dataRDD)

    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(15, computeU = true)
    val U: RowMatrix = svd.U  // The U factor is a RowMatrix.
    val s: Vector = svd.s  // The singular values are stored in a local dense vector.
    val V: Matrix = svd.V  // The V factor is a local dense matrix.
    val rows = U.rows.collect()
    val out = new PrintWriter("UserMat.txt")
    for (row<-rows){
      out.println(row.toArray.mkString(" "))
    }
    out.close()
    val out2 = new PrintWriter("Singular.txt")
    out2.println(s.toArray.mkString(" "))
    out2.close()
    val VArr = V.toArray
    val out3 = new PrintWriter("ItemMat.txt")
    for (i<-0 until V.numRows){
      for (j<-0 until V.numCols){
        out3.print(VArr(j*V.numRows+i)+" ")
      }
      out3.println()
    }
    out3.close()
  }
}
