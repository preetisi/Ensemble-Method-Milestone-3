import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.GridSearch;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.filters.AllFilter;

public class TuningClassificationRunner extends ClassificationRunner {
	private static final Map<Class<?>, String> PARAMETER_TUNING_OPTIONS;
	static {
		PARAMETER_TUNING_OPTIONS = new HashMap<Class<?>, String>();
		// PARAMETER_TUNING_OPTIONS.put(SMO.class, "C 1 20 20");
		PARAMETER_TUNING_OPTIONS.put(MultilayerPerceptron.class, "N 1 3 3");
	}

	private static final Map<Class<?>, String> GRIDSEARCH_OPTIONS;
	static {
		GRIDSEARCH_OPTIONS = new HashMap<Class<?>, String>();
		GRIDSEARCH_OPTIONS.put(SMO.class, "C RBFKernel.gamma");
	}

	public TuningClassificationRunner(String dataDir, String[] dataSets,
			Class<?>[] classifiersClasses) {
		super(dataDir, dataSets, classifiersClasses);
	}

	@Override
	protected ClassifierResult runClassifier(Instances trainingData, Instances testingData,
			Class<?> classifierClass) throws Exception {
		if (PARAMETER_TUNING_OPTIONS.containsKey(classifierClass)) {
			return runCVParameterTuning(trainingData, testingData, classifierClass,
					PARAMETER_TUNING_OPTIONS.get(classifierClass));
		} else if (GRIDSEARCH_OPTIONS.containsKey(classifierClass)) {
			return runGridSearch(trainingData, testingData, classifierClass,
					GRIDSEARCH_OPTIONS.get(classifierClass));
		} else if (classifierClass.equals(AdaBoostM1.class)) {
			return runParameterTuningAdaboost(trainingData, testingData);
		} else {
			return super.runClassifier(trainingData, testingData, classifierClass);
		}
	}

	void debugClassifiers() throws Exception {
		SortedMap<String, ClassifierResult> bestResults = new TreeMap<String, ClassifierResult>();
		for (String dataSet : this.dataSets) {
			System.out.printf("Running dataset '%s'\n", dataSet);
			Instances trainingData = getTrainingInstances(dataDir, dataSet);
			Instances testingData = getTestingInstances(dataDir, dataSet);

			// Each classifier will be run on the dataset and results stored.
			String modelPath = dataDir + "/" + dataSet + ".model";
			Classifier model = (Classifier) SerializationHelper
					.read(new FileInputStream(modelPath));
			double modelError = getError(model, trainingData, testingData);
			ClassifierResult nbResult = runClassifier(trainingData, testingData, NB_CLASSIFIER);
			ClassifierResult result = new ClassifierResult(modelError, model);
			result.setRatioResult(nbResult);
			bestResults.put(dataSet, result);
		}

		outputResults(dataDir, bestResults);
	}

	private ClassifierResult runParameterTuningAdaboost(Instances trainingData,
			Instances testingData) throws Exception {

		List<ParameterSelectedResult> resultList = new ArrayList<ParameterSelectedResult>();

		for (double param : new Double[] { 0.1, 0.2, 0.3, 0.4, 0.5 }) {
			AdaBoostM1 classifier = new AdaBoostM1();

			J48 weakLearner = new J48();
			weakLearner.setConfidenceFactor((float) param);
			classifier.setClassifier(weakLearner);

			classifier.buildClassifier(trainingData);

			double errorRate = getError(classifier, trainingData, testingData);
			resultList.add(new ParameterSelectedResult(errorRate, classifier, "C", param));
		}
		ParameterSelectedResult bestResult = Collections.min(resultList);
		bestResult.outputBestParameter();
		return bestResult;
	}

	private ClassifierResult runCVParameterTuning(Instances trainingData, Instances testingData,
			Class<?> classifierClass, String parameterTuningOptions) throws Exception {
		CVParameterSelection tuner = new CVParameterSelection();
		Classifier classifier = Classifier.forName(classifierClass.getName(), null);
		tuner.setClassifier(classifier);
		tuner.addCVParameter(parameterTuningOptions);
		tuner.setNumFolds(10);
		tuner.buildClassifier(trainingData);
		double modelError = getError(tuner, trainingData, testingData);
		ParameterSelectedResult result = new ParameterSelectedResult(modelError, tuner,
				parameterTuningOptions);
		result.outputBestParameter();
		return result;
	}

	private GridSearch getGridSearchForSMO() {

		GridSearch tuner = new GridSearch();
		// set evaluation to accuracy
		tuner.setEvaluation(new SelectedTag(GridSearch.EVALUATION_ACC, GridSearch.TAGS_EVALUATION));
		tuner.setFilter(new AllFilter());
		SMO classifier = new SMO();
		classifier.setKernel(new RBFKernel());
		tuner.setClassifier(classifier);
		tuner.setXProperty("classifier.c");
		tuner.setXMin(1);
		tuner.setXMax(16);
		tuner.setXStep(1);
		tuner.setXExpression("I");
		tuner.setYProperty("classifier.kernel.gamma");
		tuner.setYMin(-5);
		tuner.setYMax(2);
		tuner.setYStep(1);
		tuner.setYBase(10);
		tuner.setYExpression("pow(BASE,I)");
		return tuner;

	}

	private ClassifierResult runGridSearch(Instances trainingData, Instances testingData,
			Class<?> classifierClass, String gridSearchOptions) throws Exception {

		GridSearch tuner;
		if (classifierClass.equals(SMO.class)) {
			tuner = getGridSearchForSMO();
		} else {
			throw new IllegalArgumentException("unsupported class for gridsearch: "
					+ classifierClass);
		}

		tuner.buildClassifier(trainingData);

		// Output tuned parameter values.
		String xParam = gridSearchOptions.split(" ")[0];
		double xValue = tuner.getMeasure("measureX");
		String yParam = gridSearchOptions.split(" ")[1];
		double yValue = tuner.getMeasure("measureY");
		System.out.printf("Best results: %s = %f, %s = %f\n", xParam, xValue, yParam, yValue);

		double modelError = getError(tuner, trainingData, testingData);
		return new ClassifierResult(modelError, tuner);
	}

	@Override
	protected SortedMap<String, ClassifierResult> getBestResults(
			Map<String, Collection<ClassifierResult>> results) {
		// Maps classifierName -> total error ratio.
		Map<String, Double> totalErrors = new HashMap<String, Double>();

		// Count up the total error for each classifier.
		for (Entry<String, Collection<ClassifierResult>> dataResultsEntry : results.entrySet()) {
			for (ClassifierResult result : dataResultsEntry.getValue()) {
				Double currentError = totalErrors.get(result.getName());
				currentError = currentError == null ? 0 : currentError;
				currentError += result.getErrorRatio();
				totalErrors.put(result.getName(), currentError);
			}
		}

		// Find classifier with lowest total error.
		String bestClassifierName = null;
		double lowestError = Double.POSITIVE_INFINITY;
		for (Entry<String, Double> errorTotals : totalErrors.entrySet()) {
			if (errorTotals.getValue() < lowestError) {
				bestClassifierName = errorTotals.getKey();
				lowestError = errorTotals.getValue();
			}
		}

		// Create map of dataset -> classifier result using winner from above.
		SortedMap<String, ClassifierResult> bestResults = new TreeMap<String, ClassifierResult>();
		for (Entry<String, Collection<ClassifierResult>> dataResultsEntry : results.entrySet()) {
			// Find the winning classifier with the matching name.
			ClassifierResult bestResult = null;
			for (ClassifierResult result : dataResultsEntry.getValue()) {
				if (result.getName().equals(bestClassifierName)) {
					bestResult = result;
					break;
				}
			}
			if (bestResult == null) {
				throw new IllegalStateException("Didn't find a best result?");
			} else {
				bestResults.put(dataResultsEntry.getKey(), bestResult);
			}
		}
		return bestResults;
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.err.println("Usage: TuningClassificationRunner <data dir>");
		} else {
			new TuningClassificationRunner(args[0], DATASETS, CLASSIFIERS).runClassifiers();
		}
	}
}
