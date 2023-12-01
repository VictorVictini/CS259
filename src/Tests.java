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
    public static void WriteFile(int[] bayes, int[] knn, int[] testing) throws IOException {
        Assert(bayes.length == knn.length && knn.length == testing.length);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter("src\\files\\model-results.csv"))) {
            bw.write("Naive Bayes Predictions,KNN Classifier Predictions,Testing Labels\n");
            for (int i = 0; i < testing.length; i++) {
                bw.write(bayes[i] + "," + knn[i] + "," + testing[i] + "\n");
            }
            bw.close();
        }
    }
    public static void main(String[] args) {

        // knn model stuff
        knnClassifier knn = new knnClassifier();
        double[][] trainingData = new double[100][];
        int[] trainingLabels = new int[100];
        double[][] testingData = new double[100][];
        int[] testingLabels = new int[100];
        try {
            // You may need to change the path:
            knn.loadData("src\\files\\training-set.csv", trainingData, trainingLabels);
            knn.loadData("src\\files\\testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        int k = (int)Math.sqrt(trainingData.length);
        if (k % 2 == 0) k++;

        // Compute accuracy on the testing set
        int correctPredictions = 0;
        int[] knnLabels = new int[testingData.length];
        for (int i = 0; i < testingData.length; i++) {
            int label = knn.knnClassify(trainingData, trainingLabels, testingData[i], k);
            knnLabels[i] = label;
            if (testingLabels[i] == label) correctPredictions++;
        }

        double accuracy = (double) correctPredictions / testingData.length * 100;
        System.out.printf("KNN Classifier Accuracy: %.2f%%, k: %d\n", accuracy, k);

        // naive bayes model stuff
        NaiveBayes bayes = new NaiveBayes();
        try {
            bayes.loadData("src\\files\\training-set.csv", trainingData, trainingLabels);
            bayes.loadData("src\\files\\testing-set.csv", testingData, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

        NaiveBayes.NaiveBayesModel M = new NaiveBayes.NaiveBayesModel();

        // Initialising feature counts to 0s:
        for (int x = 0; x < bayes.NumberOfFeatures; x++) {
            bayes.FeatureCountsPos[x]=0;
            bayes.FeatureCountsNeg[x]=0;
        }

        // Update our feature count tables:
        for( int j = 0;j<trainingData.length; j++) {
            M.Update(trainingData[j], trainingLabels[j]); //trainingData is updating counts for
        }
        //IMPORTANT use above for model 2
        //System.out.println("testing " + M.estimate(testingData[1]));
        int[] bayesLabels = M.ReportAccuracy(testingData, testingLabels);

        try {
            WriteFile(bayesLabels, knnLabels, testingLabels);
        }
        catch (IOException e) {
            System.out.println("Error reading data files: " + e.getMessage());
            return;
        }

    }
}
