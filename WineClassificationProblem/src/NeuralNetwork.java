//Done by Taras Buchynskyi
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class NeuralNetwork {
    static Random r = new Random();

    static double numberForTest;
    static int correctOutput = 0;
    static ArrayList<double[]> inputToHidden = new ArrayList<>(); // trainingSet
    static ArrayList<double[]> inputToHiddenToTest = new ArrayList<>(); // trainingSet
    static double[] inputToOutput = new double[5];
    static double[] realOutput = new double[3];
    static ArrayList<double[]> desiredOutput = new ArrayList<>();
    static ArrayList<double[]> desiredOutputToTest = new ArrayList<>();
    static double[] hiddenNetValues = new double[5];
    static double[] outputNetValues = new double[3];

    static double[][] weightsToHidden = new double[5][13]; //[numberOfHiddenNeurons][NumberOfInputs]
    static double[][] weightsToOutput = new double[3][5];
    static double[] biasToHidden = new double[5];
    static double[] biasToOutput = new double[3];

    static double[] outputNeuronError = new double[3];
    static double[] hiddenNeuronError = new double[5];

    static HashMap<Integer, Double> costFunction = new HashMap<>();

    static int iterNumber = 200;
    static double learningRate = 0.7;
    //Taras Buchynskyi
    public static void main(String[] args) {

        readfile("./wineClassification.txt");
        //for Normalization
        ArrayList<double[]> meanAndNormalization = meanNormalization();
        int end = 0;
        for (int rows = 0; rows < weightsToHidden.length; rows++) {
            for (int columns = 0; columns < weightsToHidden[rows].length; columns++) {
                weightsToHidden[rows][columns] = randomWeights();
            }
            biasToHidden[rows] = randomWeights();
        }
        for (int rows = 0; rows < weightsToOutput.length; rows++) {
            for (int columns = 0; columns < weightsToOutput[rows].length; columns++) {
                weightsToOutput[rows][columns] = randomWeights();
            }
            biasToOutput[rows] = randomWeights();
        }

        do {
            for (int i = 0; i < inputToHidden.size(); i++) {
                hiddenNetValues = calculateNetFunction(inputToHidden.get(i), weightsToHidden, biasToHidden);
                inputToOutput = calculateHiddenActivation(hiddenNetValues);
                outputNetValues = calculateNetFunction(inputToOutput, weightsToOutput, biasToOutput);
                realOutput = calculateOutputActivation(outputNetValues);
                outputNeuronError = recalculateOutputNeuronError(realOutput, desiredOutput.get(i));
                hiddenNeuronError = recalculateHiddenNeuronError(weightsToOutput, outputNeuronError, inputToOutput);
                weightsToOutput = recalculateWeights(weightsToOutput, learningRate, outputNeuronError, inputToOutput);
                weightsToHidden = recalculateWeights(weightsToHidden, learningRate, hiddenNeuronError, inputToHidden.get(i));
                biasToHidden = recalculateBias(biasToHidden, learningRate, hiddenNeuronError);
                biasToOutput = recalculateBias(biasToOutput, learningRate, outputNeuronError);
                if (end == iterNumber - 1) {
                    calculatePercentage(realOutput, desiredOutput.get(i));
                }
            }
            ++end;
            System.out.println("-----------Learning epoch # " + end + "--------------");

        } while (end < iterNumber);


        double percentage = correctOutput * 100 / inputToHidden.size();
        System.out.println(percentage+"%");
        correctOutput = 0;
        //normalizartion for Test
        for (int columns = 0; columns < inputToHiddenToTest.get(columns).length; columns++) {
            for (int rows = 0; rows < inputToHiddenToTest.size(); rows++) {
                inputToHiddenToTest.get(rows)[columns] = (inputToHiddenToTest.get(rows)[columns] - meanAndNormalization.get(0)[columns]) / meanAndNormalization.get(1)[columns];
            }
        }
        for (int i = 0; i < inputToHiddenToTest.size(); i++) {
            hiddenNetValues = calculateNetFunction(inputToHiddenToTest.get(i), weightsToHidden, biasToHidden);
            inputToOutput = calculateHiddenActivation(hiddenNetValues);
            outputNetValues = calculateNetFunction(inputToOutput, weightsToOutput, biasToOutput);
            realOutput = calculateOutputActivation(outputNetValues);
            outputNeuronError = recalculateOutputNeuronError(realOutput, desiredOutputToTest.get(i));
            calculatePercentage(realOutput, desiredOutputToTest.get(i));
        }
        percentage = correctOutput * 100 / inputToHiddenToTest.size();
        System.out.println(percentage + "%");
    }


    private static double[] recalculateBias(double[] biasToHidden, double learningRate, double[] hiddenNeuronError) {
        for (int i = 0; i < biasToHidden.length; i++) {
            biasToHidden[i] = biasToHidden[i] + learningRate * hiddenNeuronError[i];
        }
        return biasToHidden;
    }

    private static double[][] recalculateWeights(double[][] weights, double learningRate, double[] neuronError, double[] input) {
        for (int rows = 0; rows < weights.length; rows++) {
            for (int columns = 0; columns < weights[rows].length; columns++) {
                weights[rows][columns] = weights[rows][columns] + learningRate * neuronError[rows] * input[columns];
            }
        }
        return weights;
    }

    private static double[] recalculateHiddenNeuronError(double[][] weightsToOutput, double[] outputNeuronError, double[] inputToOutput) {
        double[] derivative = new double[inputToOutput.length];
        double[] result = new double[inputToOutput.length];
        double[] errorAndWeightSum = new double[inputToOutput.length];
        for (int i = 0; i < inputToOutput.length; i++) {
            if (inputToOutput[i] >=0) derivative[i] = 1;
            else if (inputToOutput[i] < 0)  derivative[i]=0;
//            //FOR RELU

            //  derivative[i] = inputToOutput[i] * (1 - inputToOutput[i]);// FOR UNISIGMOIDAL;

        }
        for (int columns = 0; columns < weightsToOutput[1].length; columns++) {
            for (int rows = 0; rows < weightsToOutput.length; rows++) {
                errorAndWeightSum[columns] += weightsToOutput[rows][columns] * outputNeuronError[rows];
            }
            result[columns] = derivative[columns] * errorAndWeightSum[columns];
        }
        return result;
    }

    //softMax function
    private static double[] recalculateOutputNeuronError(double[] realOutput, double[] desiredOutput) {
        //show the difference between real and desired
        for (double i : realOutput)
            System.out.print(i + " ");
        System.out.println();
        for (double i : desiredOutput) System.out.print(i + " ");
        System.out.println("\n-----------");
        //
        double[] result = new double[realOutput.length];
        for (int i = 0; i < realOutput.length; i++) {
            result[i] = realOutput[i]*(1-realOutput[i]) * (desiredOutput[i] - realOutput[i]);  // For SOFTMAX
//            result[i] = realOutput[i] * (1 - realOutput[i]) * (desiredOutput[i] - realOutput[i]);// FOR UNISIGMOIDAL;
        }
        return result;
    }

    private static void calculatePercentage(double[] realOutput, double[] desiredOutput) {
        double maxInDesired = desiredOutput[0];
        int indexDesired = 0;
        double maxInReal = realOutput[0];
        int indexReal = 0;
        for (int i = 0; i < realOutput.length; i++) {
            if (maxInDesired < desiredOutput[i]) {
                maxInDesired = desiredOutput[i];
                indexDesired = i;
            }
            if (maxInReal < realOutput[i]) {
                maxInReal = realOutput[i];
                indexReal = i;
            }
        }
        if (indexReal == indexDesired) correctOutput++;
        else System.out.println("WRONG!!!");
    }

    private static double[] calculateOutputActivation(double[] outputNetValues) {
        double eSum = 0;
        double[] result = new double[outputNetValues.length];
//        for (int i = 0; i < outputNetValues.length; i++)
//            eSum += Math.pow(Math.E, outputNetValues[i]);
//        for (int i = 0; i < outputNetValues.length; i++) {
//            result[i] = Math.pow(Math.E, outputNetValues[i]) / eSum;
//        }
        //          SOFTMAX ACTIVATION


        for (int i = 0; i < outputNetValues.length; i++)
            result[i] = 1 / (1 + Math.pow(Math.E, -outputNetValues[i]));
//              UNIPOLAR SIGMOIDAL
        return result;
    }

    private static double[] calculateHiddenActivation(double[] hiddenNetValues) {
        double[] result = new double[hiddenNetValues.length];
        for (int i = 0; i < hiddenNetValues.length; i++) {
//            if (hiddenNetValues[i] >= 0) result[i] = hiddenNetValues[i];
//            else result[i] = 0;
//            ReLU


            result[i] = 1 / (1 + Math.pow(Math.E, -hiddenNetValues[i]));
//                     SIGMOIDAL
        }
        return result;
    }


    private static double[] calculateNetFunction(double[] input, double[][] weights, double[] bias) {
        double[] result = new double[weights.length];
        for (int row = 0; row < weights.length; row++) {//until reach the last neuron(row)
            for (int column = 0; column < weights[row].length; column++)
                result[row] += weights[row][column] * input[column];
            result[row] += bias[row];
        }
        return result;
    }

    private static ArrayList<double[]> meanNormalization() {
        ArrayList<double[]> result = new ArrayList<>();
        double[] mean = new double[13];
        double[] std = new double[13];

        for (int columns = 0; columns < inputToHidden.get(columns).length; columns++) {
            double tempResult = 0;
            double max = inputToHidden.get(0)[columns];
            double min = inputToHidden.get(0)[columns];
            for (int rows = 0; rows < inputToHidden.size(); rows++) {
                if (max < inputToHidden.get(rows)[columns]) max = inputToHidden.get(rows)[columns];
                if (min > inputToHidden.get(rows)[columns]) min = inputToHidden.get(rows)[columns];
                tempResult += inputToHidden.get(rows)[columns];
            }
            mean[columns] = tempResult / inputToHidden.size();
            std[columns] = max - min;
        }

        for (int columns = 0; columns < inputToHidden.get(columns).length; columns++) {
            for (int rows = 0; rows < inputToHidden.size(); rows++) {
                inputToHidden.get(rows)[columns] = (inputToHidden.get(rows)[columns] - mean[columns]) / std[columns];
            }
        }
        result.add(mean);
        result.add(std);
        return result;
    }

    public static void readfile(String fileName) {

        // This will be the output - a list of rows, each with 'width' entries:

        try {
            FileReader fileReader = new FileReader(fileName);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String forWork = bufferedReader.readLine();
            ArrayList<double[]> data = new ArrayList<>();
            double[] dataToAdd;
            int lineCounter = 0;
            while (forWork != null) {
                dataToAdd = new double[14];
                lineCounter++;
                for (int i = 0; i < dataToAdd.length; i++) {
                    dataToAdd[i] = Double.parseDouble(forWork.split(",")[i]);
                }
                data.add(dataToAdd);
                forWork = bufferedReader.readLine();
            }

            bufferedReader.close();

            Collections.shuffle(data);
            numberForTest = lineCounter * 0.75;
            Long L = Math.round(numberForTest);
            int training = L.intValue();
            int counter =0;
            for (int i = 0; i < data.size(); i++) {
                double[] arrayForTrainingData = new double[13];
                //set the result
                double[] expected = new double[3];

                if (data.get(i)[0]==1) {
                    expected[0] = 1;
                    expected[1] = 0;
                    expected[2] = 0;
                } else if (data.get(i)[0]==2) {
                    expected[0] = 0;
                    expected[1] = 1;
                    expected[2] = 0;
                } else if (data.get(i)[0]==3) {
                    expected[0] = 0;
                    expected[1] = 0;
                    expected[2] = 1;
                }
                //add the training data
                for (int j = 1; j < data.get(0).length; j++) {
                    arrayForTrainingData[j-1] = data.get(i)[j] ;
                }

                if (counter <= training) {
                    inputToHidden.add(arrayForTrainingData);
                    desiredOutput.add(expected);
                } else {

                    inputToHiddenToTest.add(arrayForTrainingData);
                    desiredOutputToTest.add(expected);
                }
                counter++;
            }

        } catch (FileNotFoundException ex) {
            System.out.println("Unable to open file: " + fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static double randomWeights() {
        double start = -1;
        double end = 1;
        double random = new Random().nextDouble();
        double result = start + (random * (end - start));
        return result;
    }
}

//Done by Taras Buchynskyi
