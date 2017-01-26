import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;




public class PartOne {
	/**main method */
	public static void main(String[] args) throws Exception{
		
		//please enter the file path when marking
		System.out.println("Please enter the arff file (T1) directory: "); 
		Scanner scan = new Scanner(System.in);
		String filepath =  scan.nextLine();//String filepath = "C:\\data mining\\T1.arff";
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

		Random seed = new Random(1); 
		data.randomize(seed);
		 
		//70% are the training set, remaining (30%) are the testing set 
		int trainSize = (int) Math.round(data.numInstances() * 0.7);
		int testSize = data.numInstances() - trainSize;
		
		Instances trainingSet = new Instances(data, 0, trainSize);
		Instances testingSet = new Instances(data, trainSize, testSize);
		
		trainingSet.setClassIndex( trainingSet.numAttributes() - 1 );
		testingSet.setClassIndex( testingSet.numAttributes() - 1 );
		
		//the algorithm I use is J48
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
			precision += evaluation.precision(1);
			recall += evaluation.recall(1);
			fmeasure +=	evaluation.fMeasure(1);	
		}
		
		System.out.println("\nThe average Accuracy :"+accuracy+" %\nThe average Precision :"+(precision*100/10)
				+" %\nThe average Recall :"+(recall*100/10)+" %\nThe average F-measure :"+(fmeasure*100/10)+" %");
		
	}
}
