package com;

import java.io.*;

/**
 * Created by zjcxj on 2016/9/1.
 */
public class textCombine {
    public static void main(String[] args){
        File file1 = new File("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\test.txt");
        File file2 = new File("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\gbdt\\leaf_test_df7_400_4.txt");
        File out = new File("C:\\Users\\zjcxj\\Desktop\\2016ByteCup\\nolabel2.txt");
        BufferedReader reader1 = null;
        BufferedReader reader2 = null;
        BufferedWriter writer = null;
        try {
            out.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            reader1 = new BufferedReader(new FileReader(file1));
            reader2 = new BufferedReader(new FileReader(file2));
            writer = new BufferedWriter(new FileWriter(out));
            String s = null;
            String ss = null;
            while ((s=reader1.readLine())!=null){
                ss = reader2.readLine();
                writer.write(s+"\t"+ss+"\n");
            }
            reader1.close();
            reader2.close();
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
