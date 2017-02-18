package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.Attribute;
import cs446.weka.classifiers.trees.Id3;
import cs446.homework2.Gradient;
import java.util.Random;

public class WekaTester {

    public static void main(String[] args) throws Exception {

        if (args.length != 5) {
            System.err.println("Usage: WekaTester arff-file");
            System.exit(-1);
        }

        // Load the data
        Instances data[] = new Instances[5];
        for (int i = 0; i < 5; i++) {
            // The last attribute is the class label
            data[i] = new Instances(new FileReader(new File(args[i])));
            data[i].setClassIndex(data[i].numAttributes() - 1);
        }

        // Cross-Validation
        Instances train[] = new Instances[5];
        Instances test[] = new Instances[5];
        for (int i = 0; i < 5; i++) {
            test[i] = new Instances(data[i]);
            // train[i] = new Instances();
            train[i] = new Instances(data[i]);
            train[i].delete();
            for (int j = 0; j < 5; j++) {
                if (i != j) {
                    for (int l = 0; l < data[j].numInstances(); l++) {
                        train[i].add(data[j].instance(l));
                    }
                }
            }
        }


        ////////////////////////////////////////////////////
        // Create a new ID3 classifier for unlimited depth//
        ////////////////////////////////////////////////////
        for (int i = 0; i < 5; i++) {
            Id3 classifier = new Id3();
            classifier.setMaxDepth(-1);
            classifier.buildClassifier(train[i]);
            // Evaluate on the test set
            Evaluation evaluation = new Evaluation(test[i]);
            evaluation.evaluateModel(classifier, test[i]);
            // Print the classifier
            System.out.println("==================================================");
            System.out.println("ID3, unlimited, fold" + (i + 1) + "\n");
            System.out.println(classifier);
            System.out.println();
            System.out.println(evaluation.toSummaryString());
        }


        /////////////////////////////////////////////
        // Create a new ID3 classifier for depth 4 //
        /////////////////////////////////////////////
        for (int i = 0; i < 5; i++) {
            Id3 classifier4 = new Id3();
            classifier4.setMaxDepth(4);
            classifier4.buildClassifier(train[i]);
            // Evaluate on the test set
            Evaluation evaluation4 = new Evaluation(test[i]);
            evaluation4.evaluateModel(classifier4, test[i]);
            // Print the classifier
            System.out.println("==================================================");
            System.out.println("ID3, depth 4, fold" + (i + 1) + "\n");
            // System.out.println(classifier4);
            System.out.println();
            System.out.println(evaluation4.toSummaryString());
        }


        /////////////////////////////////////////////
        // Create a new ID3 classifier for depth 8 //
        /////////////////////////////////////////////
        for (int i = 0; i < 5; i++) {
            Id3 classifier8 = new Id3();
            classifier8.setMaxDepth(8);
            classifier8.buildClassifier(train[i]);
            // Evaluate on the test set
            Evaluation evaluation8 = new Evaluation(test[i]);
            evaluation8.evaluateModel(classifier8, test[i]);
            // Print the classifier
            System.out.println("==================================================");
            System.out.println("ID3, depth 8, fold" + (i + 1) + "\n");
            // System.out.println(classifier8);
            System.out.println();
            System.out.println(evaluation8.toSummaryString());
        }


        /////////////////////////////////
        // Create a new SGD classifier //
        /////////////////////////////////
        for (int i = 0; i < 5; i++) {
            // Create a new classifier for SGD
            Gradient classifierSGD = new Gradient();
            classifierSGD.buildClassifier(train[i]);
            // Evaluate on the test set
            Evaluation evaluationSGD = new Evaluation(test[i]);
            evaluationSGD.evaluateModel(classifierSGD, test[i]);
            // Print the classifier
            System.out.println("==================================================");
            System.out.println("SGD, fold" + (i + 1) + "\n");
            // System.out.println(classifierSGD);
            System.out.println();
            System.out.println(evaluationSGD.toSummaryString());
        }


        ////////////////////////////////////////
        // Create a new SGD-Stumps classifier //
        ////////////////////////////////////////
        for (int i = 0; i < 5; i++) {
            Id3 stumps[] = new Id3[100];
            FastVector attributes = new FastVector(101);
            // One hundred decision stumps
            for (int j = 0; j < 100; j++) {
                // Create a new classifier for GSD with stumps
                stumps[j] = new Id3();
                stumps[j].setMaxDepth(4);
                train[i].randomize(new Random(100 * i + j));
                Instances sample = new Instances(train[i], 0, train[i].numInstances() / 2);
                stumps[j].buildClassifier(sample);
            }

            // Add features for stumps
            FastVector classes = new FastVector(2);
            classes.addElement("1");
            classes.addElement("0");
            FastVector labels = new FastVector(2);
            labels.addElement("-1");
            labels.addElement("1");

            // Add attributes
            attributes = new FastVector(101);
            for (int j = 0; j < 100; j++) {
                attributes.addElement(new Attribute("Stumps-" + j, classes));
            }
            attributes.addElement(new Attribute("Class", labels));

            // Create new train
            Instances newTrain = new Instances("newTrain", attributes, train[i].numInstances());
            newTrain.setClassIndex(newTrain.numAttributes() - 1);
            int _instances = train[i].numInstances();
            for (int j = 0; j < _instances; j++) {
                Instance ins = new Instance(newTrain.numAttributes());
                ins.setDataset(newTrain);
                // Set new features
                for (int l = 0; l < 100; l++) {
                    ins.setValue(l, stumps[l].classifyInstance(train[i].instance(j)));
                }
                String label = (train[i].instance(j).classValue() == 0.0) ? "-1" : "1";
                ins.setClassValue(label);
                newTrain.add(ins);
            }

            // Create new train
            Instances newTest = new Instances("newTest", attributes, test[i].numInstances());
            newTest.setClassIndex(newTest.numAttributes() - 1);
            _instances = test[i].numInstances();
            for (int j = 0; j < _instances; j++) {
                Instance ins = new Instance(newTest.numAttributes());
                ins.setDataset(newTest);
                // Set new features
                for (int l = 0; l < 100; l++) {
                    ins.setValue(l, stumps[l].classifyInstance(test[i].instance(j)));
                }
                String label = (test[i].instance(j).classValue() == 0.0) ? "-1" : "1";
                ins.setClassValue(label);
                newTest.add(ins);
            }

            // Create SGD classifier
            Gradient classifierSGD = new Gradient();
            classifierSGD.buildClassifier(newTrain);

            // Evaluate on the test set
            Evaluation evaluationSGD = new Evaluation(newTest);
            evaluationSGD.evaluateModel(classifierSGD, newTest);

            // Print the classifier
            System.out.println("==================================================");
            System.out.println("Decision Stumps, fold" + (i + 1) + "\n");
            // System.out.println(classifierSGD);
            System.out.println();
            System.out.println(evaluationSGD.toSummaryString());
        }

    }
}
