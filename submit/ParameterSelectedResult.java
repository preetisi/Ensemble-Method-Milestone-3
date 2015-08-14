import weka.classifiers.Classifier;
import weka.classifiers.meta.CVParameterSelection;

public class ParameterSelectedResult extends ClassifierResult {
	private String parameterName;
	private Double paramValue;

	public ParameterSelectedResult(double errorRate, Classifier model, String parameterOptions) {
		super(errorRate, model);
		this.parameterName = parameterOptions.split(" ")[0];
	}

	public ParameterSelectedResult(double errorRate, Classifier model, String parameterOptions,
			double paramValue) {
		this(errorRate, model, parameterOptions);
		this.paramValue = paramValue;
	}

	private String getParamValue() {
		if (paramValue != null) {
			return paramValue.toString();
		} else if (this.model instanceof CVParameterSelection) {
			CVParameterSelection tuningModel = (CVParameterSelection) model;
			String[] bestOptions = tuningModel.getBestClassifierOptions();
			String expectedOption = "-" + this.parameterName;

			for (int i = 0; i < bestOptions.length; i += 2) {
				String parameterName = bestOptions[i];
				String parameterValue = bestOptions[i + 1];
				if (parameterName.equals(expectedOption)) {
					// Found the parameter within the options.
					return parameterValue;
				}
			}
			throw new UnsupportedOperationException("Could not find parameter value from options");
		}
		throw new UnsupportedOperationException("Could not retrieve param value");
	}

	public void outputBestParameter() {
		System.out.printf("Best value for %s = %s\n", parameterName, getParamValue());
	}
}
