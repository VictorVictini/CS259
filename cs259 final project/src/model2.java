//in progress creating template for the project (movie dataset reading, masking, KNN for K=1)
import java.io.*;
import java.util.SplittableRandom;


public class model2 {

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
	
    

 static int NumberOfFeatures = 10; 
 static double[] toFeatureVector(double id, String genre, double runtime, double year, double imdb, double rt, double budget, double boxOffice) {	
 	
     double[] feature = new double[NumberOfFeatures]; 
     //feature[0] = id;  // We use the movie id as a numeric attribute.
    
    switch (genre) { // We also use represent each movie genre as an integer number:
     
         case "Action":  feature[0] = 1; break; 
         //case "Drama":   feature[1] = 0; break;
         //case "Romance": feature[2] = 0; break;
         case "Sci-Fi": feature[1] = 1; break;
         case "Adventure": feature[2] = 1; break;
         case "Horror": feature[3] = 1; break;
         case "Mystery": feature[4] = 1; break;
         case "Thriller": feature[5] = 1; break;  
         //default: Assert(false);
     }
    //switch()
    
 
    
     if (runtime >= 109) {//109
    	 feature[6] = 1;
     }else {
    	 feature[6] = 0;
     }
     if(budget >= 98.04) { //98.04
    	 feature[7] = 1;
     }else {
    	 feature[7] = 0;
     }
     
     
     // That is all. We don't use any other attributes for prediction.
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

		
		for (int x = 0; x < NumberOfFeatures; x++) { // Update the score s according to the Naive Bayes formula for the odds: log O(C|x1,x2,x3,...) = log O(C) + log s1 + log s2 + log s3,
			// where s1 = P(x1|C)/P(x1|~C), s2 = P(x2|C)/P(x2|~C), s3 = P(x3|C)/P(x3|~C) are "feature strengths".    
			
	   		if(X[x] > 0) { // To simplify the model and to apply it later to MNIST, we only consider feature presence, not absences   
	   			
	   			 	   			
	   			double p_feature_cond_pos = FeatureCountsPos[x]/like_count; // P(x|C) = #{x AND pos} / #{pos} 
	   			if (p_feature_cond_pos == 0)
	   				p_feature_cond_pos = .01; // We make each estimated probability to be at least 0.01 to avoid division by 0 later.
	   										  // This is called "smoothing."
	   			double p_feature_cond_neg = FeatureCountsNeg[x]/dislike_count; // P(x|~C) = #{x AND ~pos} / #{~pos}	 	   			
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
	       if (estimate(data[j])   >= .643) // We apply .643 probability threshold to predict the class, when estimate is above this number, we predict yes, else we predict no
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




