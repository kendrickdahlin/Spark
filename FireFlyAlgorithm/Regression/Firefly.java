package FireFlyAlgorithm.Regression;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import javax.crypto.interfaces.PBEKey;

import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.FireflyModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;


public class Firefly {
    public static double objectiveFunction(double[] firefly, double[][] X, int[] y){
        int[] pred = predict(firefly, X);
        double mse = 0;

        for (int i = 0; i < pred.length; i++) {
            mse +=Math.pow(y[i]-pred[i],2);
        }
        return mse/y.length;
    }

    public static double[] firefly(double[][] X, int[] y){
        int n_fireflies = 50;
        double max_iter = 100;
        double gamma = 0.5;
        double delta = 0.7;
        double lb = -5;
        double ub = 5;
        int dim = X[0].length +1;

        // initialize fireflies
        double[][] fireflies = new double[n_fireflies][dim];
        for (int i = 0; i < n_fireflies; i++){
            for (int j = 0; j < dim; j++){
                fireflies[i][j] = lb + Math.random() * (ub - lb);
            }
        }
        //set fitness
        double[] fitness = new double[n_fireflies];
        for (int i = 0; i <n_fireflies; i++){
            fitness[i] = objectiveFunction(fireflies[i],X,y);
        }

        //set global bests
        int bestIndex = 0;
        for (int i = 1; i <n_fireflies; i++){
            if (fitness[i] < fitness[bestIndex]){
                bestIndex = i;
            }
        }
        double[] bestFirefly = fireflies[bestIndex];
        double bestFitness = fitness[bestIndex];

        for (int k = 0; k<max_iter; k++){
            for (int i = 0; i < n_fireflies; i++) {
                double[] pBestFirefly = fireflies[i];
                double pBestAttractiveness = 0;
                for (int j = 0; j < n_fireflies; j++) {
                    if (fitness[j] < fitness[i]){
                        double r = distance(fireflies[j], fireflies[i]); //distance squared
                        double beta1 = Math.exp(-gamma *r);
                        if (beta1> pBestAttractiveness){
                            pBestAttractiveness = beta1;
                            pBestFirefly = fireflies[j];
                        }

                    }
                }
                for (int l = 0; l < dim; l++){
                    fireflies[i][l] += delta * (pBestFirefly[l] - fireflies[i][l]);
                }
                fitness[i] = objectiveFunction(fireflies[i], X, y);
                
                if (fitness[i] < bestFitness){
                    bestFitness = fitness[i];
                    bestFirefly = fireflies[i];
                }

            }
        }
        return bestFirefly;
    }

    private static double distance(double[] arr1, double[] arr2){
        double dist = 0.0;
        for (int i = 0; i < arr1.length; i++) {
            dist+= Math.pow(arr1[i]-arr2[i],2);
        }
        return dist;
    }
    public static int[] predict(double[] model, double[][] X){
        int[] pred = new int[X.length];
        for (int i = 0; i < pred.length; i++) {
            double dot = 0;
            for (int j = 0 ; j<X[i].length; j++){
                dot += X[i][j] * model[j];
            }
            pred[i]= (dot + model[model.length-1])>=0 ? 1 :0;
        }
        return pred;
    }

    public static int[] label_encode(double[] y){
        Map<Object, Integer> classToIndex = new HashMap<>();
        int index = 0;
        for (Object c: y){
            if (!classToIndex.containsKey(c)){
                classToIndex.put(c,index++);
            }
        }
        int[] yEncoded = new int[y.length];
        for (int i = 0; i < y.length; i++){
            yEncoded[i] = classToIndex.get(y[i]);
        }
        return yEncoded;
    }

    public static double[][] standardize(double[][] X) {
        double[] mean = new double[X[0].length];
        double[] std = new double[X[0].length];
        for (int j = 0; j < X[0].length; j++) {
            double sum = 0;
            for (int i = 0; i < X.length; i++) {
                sum += X[i][j];
            }
            mean[j] = sum / X.length;
            sum = 0;
            for (int i = 0; i < X.length; i++) {
                sum += Math.pow(X[i][j] - mean[j], 2);
            }
            std[j] = Math.sqrt(sum / X.length);
        }
        double[][] X_scaled = new double[X.length][X[0].length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                X_scaled[i][j] = (X[i][j] - mean[j]) / std[j];
            }
        }
        return X_scaled;
    }


    public static double accuracy_score(int[] y_true, int[] y_pred){
        int correct = 0;
        for (int i = 0; i < y_true.length; i++) {
            if (y_true[i] == y_pred[i]){
                correct++;
            }
        }
        return (double) correct/y_true.length;
    }
    public static void main(String[] args) {
        String fileName = "Behavior.csv";
        SparkSession spark = SparkSession.builder()
                .appName("Firefly Algorithm with Spark")
                .getOrCreate();
        
        // read data
        Dataset<Row> df = spark.read()
                .csv(fileName)
                .option("header", true)
                .option("inferSchema", true);

        List<Row> featureRows = df.select(df.columns()[0], df.columns()[df.columns().length - 2])
                .collectAsList();

        double[][] X = new double[featureRows.size()][2]; // Assuming there are 2 columns selected
        for (int i = 0; i < featureRows.size(); i++) {
            X[i][0] = featureRows.get(i).getDouble(0);
            X[i][1] = featureRows.get(i).getDouble(1);
        }
        
        double[] y = df.select(df.columns[df.columns.length - 1])
                .as("label")
                .map(row -> row.getAs("label"))
                .collect();
        
        // transform y values to ints
        int[] y_encode = label_encode(y);
        
        // scale X values
        X = standardize(X);
        double[] model = firefly(X, y_encode);
        int[] yPred = predict(model, X);
    
        double accuracy = accuracy_score(y_encode,yPred);
        
        System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
        
        spark.stop();
    }
}
