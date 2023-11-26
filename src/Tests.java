//in progress creating template for the project (movie dataset reading, masking, KNN for K=1)
import java.io.*;
import java.util.*;

public class Tests {

    // Use we use 'static' for all methods to keep things simple, so we can call those methods main

    static void Assert (boolean res) // We use this to test our results - don't delete or modify!
    {
        if(!res)	{
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }

    // Copy your vector operations here:
    // ...
    static double dot(double[] U, double[] V) {
        Assert(U.length == V.length);
        double res = 0;
        for (int i = 0; i < U.length; i++) {
            res += U[i] * V[i];
        }
        return res;
    }


    static int NumberOfFeatures = 9;
    static double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice) {

        double[] feature = new double[NumberOfFeatures];
        //feature[0] = id;  // don't need this, all unique and creates data bias

        // one-hot encoding for genre
        switch (genre) {
            case "Action":    feature[0] = 7; break;
            case "Drama":     feature[0] = 1; break;
            case "Romance":   feature[0] = 3; break;
            case "Sci-Fi":    feature[0] = 6; break;
            case "Adventure": feature[0] = 4; break;
            case "Horror":    feature[0] = 2; break;
            case "Mystery":   feature[0] = 0; break;
            case "Thriller":  feature[0] = 5; break;
            default:          Assert(false);
        }
        feature[0] -= 3.8;
        feature[0] /= 3;
        feature[1] = year - 2024;
        feature[1] /= 3; // update features later

        //feature[8] = Math.log10(runtime); // subtract avg value
        // That is all. We don't use any other attributes for prediction.
        return feature;
    }

    // We are using the dot product to determine similarity:
    static double similarity(double[] u, double[] v) {
        return dot(u, v);
    }

    // We have implemented KNN classifier for the K=1 case only. You are welcome to modify it to support any K
    static int knnClassify(double[][] trainingData, int[] trainingLabels, double[] testFeature, int k) {
        Assert(1 <= k && k <= trainingLabels.length);
        ArrayList<Double> sim = new ArrayList<Double>();
        ArrayList<Double> copy = new ArrayList<Double>(); // used for an unchanged copy of sim, for sorting index
        ArrayList<Integer> index = new ArrayList<Integer>();
        for (int i = 0; i < trainingData.length; i++) {
            double s = similarity(testFeature, trainingData[i]);
            sim.add(s);
            copy.add(s);
            index.add(i);
        }
        Collections.sort(sim, (s1, s2) -> (int)Math.signum(s2 - s1)); // descending order
        Collections.sort(index, (i1, i2) -> (int)Math.signum(copy.get(i2) - copy.get(i1))); // sorted identical to sim, because parallel arrays
        int likeCount = 0;
        for (int i = 0; i < k; i++) {
            if (trainingLabels[index.get(i)] == 1) likeCount++;
        }
        return likeCount > k / 2 ? 1 : 0;
    }

    static void loadData(String filePath, double[][] dataFeatures, int[] dataLabels) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int idx = 0;
            br.readLine(); // skip header line
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                // Assuming csv format: MovieID,Title,Genre,Runtime,Year,Lead Actor,Director,IMDB,RT(%),Budget,Box Office Revenue (in million $),Like it
                double id = Double.parseDouble(values[0]);
                String genre = values[2];
                double runtime = Double.parseDouble(values[3]);
                double year = Double.parseDouble(values[4]);
                double imdb = Double.parseDouble(values[7]);
                double rt = Double.parseDouble(values[8]);
                double budget = Double.parseDouble(values[9]);
                double boxOffice = Double.parseDouble(values[10]);

                dataFeatures[idx] = toFeatureVector(id, genre, runtime, year, imdb, rt, budget, boxOffice);
                dataLabels[idx] = Integer.parseInt(values[11]); // Assuming the label is the last column and is numeric
                idx++;
            }
        }
    }

    public static void main(String[] args) {

        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try {
            // You may need to change the path:
            loadData("src\\files\\training-set.csv", trainingData, trainingLabels);
            loadData("src\\files\\testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        int k = (int)Math.sqrt(trainingData.length);
        if (k % 2 == 0) k++;

        // Compute accuracy on the testing set
        int correctPredictions = 0;

        for (int i = 0; i < testingData.length; i++) {
            int label = knnClassify(trainingData, trainingLabels, testingData[i], k);
            if (testingLabels[i] == label)
                correctPredictions++;
        }

        double accuracy = (double) correctPredictions / testingData.length * 100;
        System.out.printf("A: %.2f%%, k: %d\n", accuracy, k);

    }
}
