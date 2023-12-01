
import java.io.*;
import java.util.SplittableRandom;


public class model3 {

	// Use we use 'static' for all methods to keep things simple, so we can call those methods main

	static void Assert (boolean res) // We use this to test our results - don't delete or modify!
	{
	 if(!res)	{
		 System.out.print("Something went wrong.");
	 	 System.exit(0);
	 }
	}

	
    

 static int NumberOfFeatures = 11; 
 static double[] toFeatureVector(double id, String genre, double runtime, double year, String director, double imdb, double rt, double budget, double boxOffice) {	
 	
     double[] feature = new double[NumberOfFeatures];
    
    switch (genre) { // We also use represent each movie genre as an integer number
         case "Adventure": feature[0] = 1; break;
         case "Mystery": feature[1] = 1; break; 
         
     }
     if (runtime >= 109) {// 109
    	 feature[2] = 1;
     }
     /*if(runtime <= 109) {
    	 feature[8] = 1;
     }*/
     if(budget >= 98.04) {
    	 feature[3] = 1;
     }
     
     switch (director) {
 
	 	case "Wes Anderson": feature[5] = 1; break;
	 	case "Denis Villeneuve": feature[6] = 1; break;
	 	case "Pedro Almodóvar": feature[7] = 1; break;
 
     }
     
     
    
     return feature;
 }
 static  int like_count = 0, dislike_count = 0;
 static double [] FeatureCountsPos = new double [NumberOfFeatures];  
 static double [] FeatureCountsNeg = new double [NumberOfFeatures];
 static class NaiveBayesModel {     
		
	   public NaiveBayesModel(){ }  

	   double estimate(double [] X){  // Returns the probability of the datapoint with the features X to belong to the positive class C, here C = having a flu 
		   							// Implements the Naive Bayes prediction model as the slides 23-26 explain.
			      	
		
		double s = Math.log((double)like_count/dislike_count); 
		Assert(dislike_count > 0);
		Assert(like_count > 0);

		
		for (int x = 0; x < NumberOfFeatures; x++) {     
			
	   		if(X[x] > 0) { // To simplify the model and to apply it later to MNIST, we only consider feature presence, not absences   
	   			
	   			 	   			
	   			double p_feature_cond_pos = FeatureCountsPos[x]/like_count; 
	   			if (p_feature_cond_pos == 0)
	   				p_feature_cond_pos = .01; // We make each estimated probability to be at least 0.01 to avoid division by 0 later.
	   										  // This is called "smoothing."
	   			double p_feature_cond_neg = FeatureCountsNeg[x]/dislike_count;  	   			
	   			if (p_feature_cond_neg == 0)
	   				p_feature_cond_neg = .01; 
	   			
	   			double feature_strength = p_feature_cond_pos / p_feature_cond_neg; //  
	   			
	   			s = s + Math.log(feature_strength);
	   		}	   			   		
	   	}
		return 1/(1 + Math.exp(-s)); // Convert back from log O(C|X) to P(C|X)   
	   }

public void Update(double X[], int label) { 
	
		   // Update the tables with occurrence counts for the features:
	   Assert(NumberOfFeatures == X.length);
	   for (int x = 0; x < NumberOfFeatures; x++) { 
		   
		   if(label > 0) { 
			   if(X[x] > 0)
				   FeatureCountsPos[x] ++;
		   }
		   else  
			   if(X[x] > 0)
				   // add a line of code here!			   
				   FeatureCountsNeg[x] ++;
	   }		   
	   // Update the counts of liking and disliking movies
		   if(label > 0)  
			   like_count++;
		   	
		   else  
			   dislike_count++;
		   
}
public void ReportAccuracy(double data[][], int labels[]) {   
    double number_correct_predictions = 0;
    for(int j = 0;j<data.length; j++) {
  	  
  	  System.out.println(estimate(data[j]));
  	   int prediction;
	       if (estimate(data[j])   >= .71) // We apply .71 probability threshold to predict the class, when estimate is above this number, we predict yes, else we predict no
	    	   prediction = 1;     
	       else
	    	   prediction = 0;
  	  
        if (prediction == labels[j])
        	   number_correct_predictions++; 
        		System.out.println("number of correct predictions: " + number_correct_predictions + " label: " + labels[j]);
        		
    }
    System.out.print((double)(number_correct_predictions/data.length * 100)); 
}

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
             String director = values[6];
             double imdb = Double.parseDouble(values[7]);                
             double rt = Double.parseDouble(values[8]);  
             double budget = Double.parseDouble(values[9]);  
             double boxOffice = Double.parseDouble(values[10]);  
             
             dataFeatures[idx] = toFeatureVector(id, genre, runtime, year, director, imdb, rt, budget, boxOffice);
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
         loadData("C:\\Users\\yonip\\eclipse-workspace\\cs259 final project\\Files\\training-set.csv", trainingData, trainingLabels);            
         loadData("C:\\Users\\yonip\\eclipse-workspace\\cs259 final project\\Files\\testing-set.csv", testingData, testingLabels);            
     } 
     catch (IOException e) {
         System.out.println("Error reading data files: " + e.getMessage());
         return;
     }
     
     NaiveBayesModel M = new NaiveBayesModel();  
     

     // Initialising feature counts to 0s:
	for (int x = 0; x < NumberOfFeatures; x++) {
				   FeatureCountsPos[x]=0;
				   FeatureCountsNeg[x]=0;
	}
	
	// Update our feature count tables:
	for( int j = 0;j<trainingData.length; j++) {
		 M.Update(trainingData[j], trainingLabels[j]); //trainingData is updating counts for 
	}
	//IMPORTANT use above for model 2
	 //System.out.println("testing " + M.estimate(testingData[1]));
	 M.ReportAccuracy(testingData, testingLabels);
 }

}




