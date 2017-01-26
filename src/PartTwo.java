import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddCluster;
import weka.filters.unsupervised.attribute.Remove;


public class PartTwo {
	public static void main(String[] args) throws Exception{
		int numOfCluster = 0;
		
		//please enter the file path when marking
		System.out.println("Please enter the arff file (T2) directory: "); 
		Scanner scan = new Scanner(System.in);
		String filepath =  scan.nextLine();//String filepath = "C:\\data mining\\T2.arff";
		scan.close();
		
		BufferedReader br = null;
		br = new BufferedReader(new FileReader(filepath));
		
		Instances data = new Instances(br);
		br.close();
				
		//deleting the PatientID
		String[] options = weka.core.Utils.splitOptions("-R 1");
		Remove remove = new Remove();
		remove.setOptions(options);
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);
		
		// cluster the 8 recommended treatments
		AddCluster kmean = new AddCluster();
		options = weka.core.Utils.splitOptions("-W \"weka.clusterers.SimpleKMeans -N" + 
				" 256 -A \\\"weka.core.EuclideanDistance -R first-last\\\" -I 500 -S " + 256 + "\" -I 1-12");
		kmean.setOptions(options);
		kmean.setInputFormat(data);
		data = Filter.useFilter(data, kmean);
		
		//deleting the 8 treatments, since the clusters have added
		options = weka.core.Utils.splitOptions("-R 13-20");
		remove.setOptions(options);
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);
		
		numOfCluster = data.attribute(data.numAttributes()-1).numValues();// get how many cluster are generated
		
		Random seed = new Random(1); 
		data.randomize(seed);
		 
		int trainSize = (int) Math.round(data.numInstances() * 0.7);
		int testSize = data.numInstances() - trainSize;
		
		Instances trainingSet = new Instances(data, 0, trainSize);
		Instances testingSet = new Instances(data, trainSize, testSize);
		
		trainingSet.setClassIndex( trainingSet.numAttributes() - 1 );
		testingSet.setClassIndex( testingSet.numAttributes() - 1 );
		
		//still using J48 as the model in part two
		J48 j48 = new J48();
		j48.buildClassifier(trainingSet);
		
		Evaluation evaluation = new Evaluation(trainingSet);
		
		evaluation.evaluateModel(j48, testingSet);
		System.out.println("The number of Training instance: " + trainSize+"\nThe number of Testing instance: " + testSize);
		System.out.println(evaluation.toSummaryString("=== Summary ===", true));
		System.out.println(evaluation.toClassDetailsString());
		System.out.println(evaluation.toMatrixString());
		
		System.out.println("True positive : "+evaluation.truePositiveRate(1)+"\nTrue negative : " +evaluation.trueNegativeRate(1)
				+"\nFalse positive : "+evaluation.falsePositiveRate(1)+"\nFalse negative : "+evaluation.falseNegativeRate(1));

		double accuracy =0.0;
		double precision =0.0;
		double recall =0.0;
		double fmeasure =0.0;
		
		//get the average result of accuracy, precision, recall and F-measure
		for(int i= 1; i<=10; i++){
			trainSize = (int) Math.round(data.numInstances() * 0.7);
			testSize = data.numInstances() - trainSize;
			
			trainingSet = new Instances(data, 0, trainSize);
			testingSet = new Instances(data, trainSize, testSize);
			
			trainingSet.setClassIndex( trainingSet.numAttributes() - 1 );
			testingSet.setClassIndex( testingSet.numAttributes() - 1 );

			j48.buildClassifier(trainingSet);
			
			evaluation = new Evaluation(trainingSet);
			
			evaluation.evaluateModel(j48, testingSet);
			
			accuracy =+ evaluation.pctCorrect();
			
			for(int n = 0; n<numOfCluster;n++){
				precision += evaluation.precision(n);
				recall += evaluation.recall(n);
				fmeasure +=	evaluation.fMeasure(n);	
			}
		}
		
		System.out.println("\nThe average Accuracy :"+accuracy+" %\nThe average Precision :"+(precision*100/(numOfCluster*10))
				+" %\nThe average Recall :"+(recall*100/(numOfCluster*10))+" %\nThe average F-measure :"
		+(fmeasure*100/(numOfCluster*10))+" %");

	}
}
