package com;

/**
 * Created by zjcxj on 2016/9/25.
 */
public class Random {
    public static void main(String[] args){
        int split = 1000000;
        int sample = 100000000;
        int sum = 0;
        for (int i = 0; i<sample;i++){
            int x = (int)(Math.random()*(split));
            int y = (int)(Math.random()*(split));
            double xx = x*1.0/split;
            double yy = y*1.0/split;

            if (xx*xx+yy*yy<=1){
                sum = sum + 1;
            }
        }
        System.out.println(sum*4.0/sample);
    }
}
