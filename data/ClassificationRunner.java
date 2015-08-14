import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassificationRunner {
	final static String[] DATASETS = { "anneal", "audiology", "autos", "balance-scale",
			"breast-cancer", "colic", "credit-a", "diabetes", "glass", "heart-c", "hepatitis",
			"hypothyroid" };
	final static Class<?>[] CLASSIFIERS = { SMO.class, MultilayerPerceptron.class, AdaBoostM1.class };
	final static Class<? extends Classifier> NB_CLASSIFIER = NaiveBayes.class;

	final static String[] EXAMPLE_DATASET = { "anneal" };
	final static Class<?>[] EXAMPLE_CLASSIFIER = { SMO.class };

	final String dataDir;
	final String[] dataSets;
	final Class<?>[] classifierClasses;

	public ClassificationRunner(String dataDir, String[] dataSets, Class<?>[] classifiersClasses) {
		this.dataDir = dataDir;
		this.dataSets = dataSets;
		this.classifierClasses = classifiersClasses;
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.err.println("Usage: RunClassifiers <data dir>");
		} else {
			ClassificationRunner runner = new ClassificationRunner(args[0], DATASETS, CLASSIFIERS);
			runner.runClassifiers();
		}
	}

	static String formatClassiferName(String fullClassifierName) {
		String[] nameParts = fullClassifierName.split("\\.");
		return nameParts[nameParts.length - 1];
	}

	static void runExampleClassifier(String dataDir) throws Exception {
		ClassificationRunner runner = new ClassificationRunner(dataDir, EXAMPLE_DATASET,
				EXAMPLE_CLASSIFIER);
		runner.runClassifiers();
	}

	void runClassifiers() throws Exception {
		SortedMap<String, Collection<ClassifierResult>> results = new TreeMap<String, Collection<ClassifierResult>>();
		for (String dataSet : this.dataSets) {
			System.out.printf("Running dataset '%s'\n", dataSet);
			Instances trainingData = getTrainingInstances(dataDir, dataSet);
			Instances testingData = getTestingInstances(dataDir, dataSet);

			ClassifierResult nbResult = runClassifier(trainingData, testingData, NB_CLASSIFIER);

			// Each classifier will be run on the dataset and results stored.
			List<ClassifierResult> classifierResults = new ArrayList<ClassifierResult>(
					classifierClasses.length);

			for (Class<?> classifierClass : this.classifierClasses) {
				System.out.printf("\tRunning classifier: %s...\n", classifierClass.getSimpleName());

				ClassifierResult result = runClassifier(trainingData, testingData, classifierClass);
				result.setRatioResult(nbResult);
				classifierResults.add(result);
			}
			results.put(dataSet, classifierResults);
		}

		outputResults(dataDir, getBestResults(results));
	}

	static Instances getTrainingInstances(String dataDir, String dataSet) throws Exception {
		return getInstances(dataDir, dataSet, true/* isTraining */);
	}

	static Instances getTestingInstances(String dataDir, String dataSet) throws Exception {
		return getInstances(dataDir, dataSet, false/* isTraining */);
	}

	/**
	 * Returns the instances for a dataset.
	 * 
	 * @param dataDir
	 *            the root directory containing the dataset arff file
	 * @param dataSet
	 *            the name of the dataset
	 * @param isTraining
	 *            whether this is a training or testing dataset.
	 */
	static Instances getInstances(String dataDir, String dataSet, boolean isTraining)
			throws Exception {
		String datasetRootName = dataDir + "/" + dataSet;
		String trainingDataPath = datasetRootName + (isTraining ? "_train.arff" : "_test.arff");

		DataSource inputData = new DataSource(trainingDataPath);

		Instances instances = inputData.getDataSet();

		if (instances.classIndex() == -1) {
			instances.setClassIndex(instances.numAttributes() - 1);
		}

		return instances;
	}

	static String getModelOutputPath(String dataDir, String dataSet) {
		String datasetRootName = dataDir + "/" + dataSet;
		return datasetRootName + ".model";
	}

	static void outputSummaryResults(Collection<ClassifierResult> bestResults) {
		ClassifierResult maxResult = Collections.max(bestResults);
		System.out.println("Max error ratio: " + maxResult.getErrorRatio());

		// Find the mean error rate.
		double totalErrorRatio = 0;
		for (ClassifierResult result : bestResults) {
			totalErrorRatio += result.getErrorRatio();
		}

		double meanErrorRatio = (totalErrorRatio / bestResults.size());
		System.out.println("Mean error ratio: " + meanErrorRatio);
	}

	static double getError(Classifier model, Instances trainData, Instances testData)
			throws Exception {
		// Evaluate on the test dataset.
		Evaluation eval = new Evaluation(trainData);
		eval.evaluateModel(model, testData);
		return eval.errorRate();
	}

	protected void outputResults(String outputDir, SortedMap<String, ClassifierResult> results)
			throws FileNotFoundException, IOException {
		System.out.println("Outputting all models to " + outputDir);

		for (Entry<String, ClassifierResult> dataResultsEntry : results.entrySet()) {
			String dataSet = dataResultsEntry.getKey();
			ClassifierResult bestResult = dataResultsEntry.getValue();

			// Output min result.
			System.out.printf("Best error ratio: %f (%s)\n", bestResult.getErrorRatio(),
					bestResult.getName());
			bestResult.outputModel(getModelOutputPath(outputDir, dataSet));
		}

		outputSummaryResults(results.values());
	}

	/**
	 * Get the best result for each dataset.
	 */
	protected SortedMap<String, ClassifierResult> getBestResults(
			Map<String, Collection<ClassifierResult>> results) {
		SortedMap<String, ClassifierResult> bestResults = new TreeMap<String, ClassifierResult>();
		for (Entry<String, Collection<ClassifierResult>> dataResultsEntry : results.entrySet()) {
			Collection<ClassifierResult> dataResults = dataResultsEntry.getValue();
			// For each dataset, get the result with the lowest error rate.
			bestResults.put(dataResultsEntry.getKey(), Collections.min(dataResults));
		}
		return bestResults;
	}

	protected ClassifierResult runClassifier(Instances trainingData, Instances testingData,
			Class<?> classifier) throws Exception {
		Classifier model = Classifier.forName(classifier.getName(), null);
		model.buildClassifier(trainingData);

		double modelError = getError(model, trainingData, testingData);
		return new ClassifierResult(modelError, model);
	}

}
