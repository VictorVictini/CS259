import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class knnClassifier {
    // Use we use 'static' for all methods to keep things simple, so we can call those methods main
    static void Assert (boolean res) // We use this to test our results - don't delete or modify!
    {
        if(!res)	{
            System.out.print("Something went wrong.");
            System.exit(0);
        }
    }

    // vector operation used:
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

        // one-hot encoding for genre
        switch (genre) {
            case "Action":    feature[0] = 1; break;
            case "Drama":     feature[1] = 1; break;
            case "Romance":   feature[2] = 1; break;
            case "Sci-Fi":    feature[3] = 1; break;
            case "Adventure": feature[4] = 1; break;
            case "Horror":    feature[5] = 1; break;
            case "Mystery":   feature[6] = 1; break;
            case "Thriller":  feature[7] = 1; break;
            default:          Assert(false);
        }

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
}
