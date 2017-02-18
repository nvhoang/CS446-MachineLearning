package cs446.homework2;

import weka.classifiers.*;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Instances;
import weka.core.Instance;
import java.util.Random;

public class Gradient extends Classifier {

    private double _rate = 0.00001;
    private double _threshold = 100;
    private double _step = 0.0001;
    private double[] _weight;
    private double _bias = 0.0;
    private int _attributes;
    private int _instances;

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Builds SDG classifier.
     *
     * @param data the training data
     * @exception Exception if classifier can't be built successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        // build SGD classifier
        makeSGD(data);
    }

    /**
     * Builds SDG classifier.
     *
     * @param data the training data
     * @exception Exception if classifier can't be built successfully
     */
    private void makeSGD(Instances data) throws Exception {
        // Initialization
        _instances = data.numInstances();
        _attributes = data.numAttributes() - 1;
        _weight = new double[_attributes];
        for (int i = 0; i < _attributes; i++) {
            _weight[i] = 0.0;
        }

        // Check if no instances have reached this node.
        if (_instances == 0) {
            throw new Exception("No Instances.");
        }

        // Calculate cost
        double cost = calculateCost(data);
        double prev = 0.0;
        long seed = 667244960;
        while (cost > _threshold && Math.abs(cost - prev) > _step) {
            // Randomly shuffle training examples
            seed += 1;
            Random random = new Random(seed);
            data.randomize(random);
            for (int i = 0; i < _instances; i++) {
                Instance ins = data.instance(i);
                double y = predict(ins);
                double x = sign(ins.classValue());
                for (int j = 0; j < _attributes; j++) {
                    _weight[j] -= _rate * (y - x) * (1.0 - ins.value(j));
                }
                _bias -= _rate * (y - x);
            }
            prev = cost;
            cost = calculateCost(data);
        }
    }

    /**
     * Binary classifier sign()
     *
     * @param val
     */
    public double sign(double val) {
        double label = (val == 0.0) ? -1.0 : 1.0;
        return label;
    }

    /**
     * Calculate predicted label y
     *
     * @param data the training data
     */
    private double predict(Instance ins) {
        double sum = 0.0;
        for (int i = 0; i < _attributes; i++) {
            sum += (1.0 - ins.value(i)) * _weight[i];
        }
        sum += _bias;
        double label = (sum >= 0.0) ? 1.0 : -1.0;
        return label;
    }

    /**
     * Calculate cost: 1/2(y-w*x)^2
     *
     * @param data the training data
     */
    private double calculateCost(Instances data) {
        double cost = 0.0;
        double y, x;
        for (int i = 0; i < _instances; i++) {
            Instance ins = data.instance(i);
            cost += Math.pow(predict(ins) - sign(ins.classValue()), 2);
        }
        return cost / 2.0;
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    @Override
    public double classifyInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException(
                    "ID3: no missing values, " + "please.");
        }
        double label = predict(instance);
        return (label == 1.0) ? 1.0 : 0.0;
    }
}
