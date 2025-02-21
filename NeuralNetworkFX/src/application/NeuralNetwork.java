/*
 * 
 * 
 * neural network object
 * creates and manages a neural network.
 * 
 * 
 * */


/*
		UPDATED STRUCTURE:

	Weights:
		The weights are now stored in an array of matrices:
		- L_i = layers
		- N_i = neurons
		- W_i = weights
		
	Biases:
		The biases are now stored in an array of arrays.
		
	Activations:
		Activated neuron outputs (activations) are now stored in an array of arrays.
		
	Outputs:
		Pre-activated neuron outputs (plain output or Z) are now stored in an array of arrays. 
		// Used for applying the chain rule
		
	Architecture:
		The architecture is now defined by a vector, where each element represents a layer and its value indicates the number of neurons in that layer.
		Example:
			Int[] arch = {2,10,10,5,1};
			This represents the following architecture:
			- 2 neurons in the input layer
			- 10 neurons in the first hidden layer
			- ...
			- 1 neuron in the output layer
			For a total of 5 layers (arch.size).
	
	Arrays of arrays/matrices are used instead of directly using matrices or 3D matrices 
	to accommodate layers and neuron weights of different sizes dynamically. 
	This allows each layer to have a dynamically allocated matrix based on the neural network's architecture.
	
	Organizing weights and biases into arrays and matrices facilitates implementation for parallel computations. 
	Libraries like OpenMP (for multithreading) or CUDA (for GPU acceleration) can efficiently handle these matrix operations.
	
	Matrices:
		- N (first value) = columns
		- M (second value) = rows
		
		The first matrix in the array is initialized with n = 1 and m = architecture[0], 
		as one input corresponds to one weight and one bias. Each row represents a neuron.
		Other matrices are initialized with dimensions architecture[i-1] × architecture[i].
*/



package application;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;


public class NeuralNetwork implements Serializable{

	private static final long serialVersionUID = 972856922459233840L;
	
	int[] architecture;
    Matrix[] weights;
    Matrix[] biases;
    Matrix[] weightsGradients;
    Matrix[] biasesGradients;
    Matrix[] inputGradients;
    Matrix[] outputs;
    Matrix[] activations;
    int layerCount;
    String hiddenLayersAF;
    String outputLayerAF;
    String lossFunction;
    private double learningRate;
    int numOutputs;
    
	public NeuralNetwork(int[] architecture) {
		super();
		this.architecture = architecture;
		this.layerCount = architecture.length;
		this.weights = new Matrix[layerCount];
		this.biases = new Matrix[layerCount];
		this.weightsGradients = new Matrix[layerCount];
		this.biasesGradients = new Matrix[layerCount];
		this.inputGradients = new Matrix[layerCount];
		this.activations = new Matrix[layerCount];
		this.outputs = new Matrix[layerCount];
		this.numOutputs = architecture[architecture.length-1];
		
		weights[0] = new Matrix(1, architecture[0]);
	    biases[0] = new Matrix(1, architecture[0]);
	    
	    weightsGradients[0] = new Matrix(1, architecture[0]);
	    biasesGradients[0] = new Matrix(1, architecture[0]);
	    inputGradients[0] = new Matrix(1, architecture[0]);
	    
	    outputs[0] = new Matrix(1, architecture[0]);
	    activations[0] = new Matrix(1, architecture[0]);
	    
	    initializeMatrix(weights[0], 1);
	    initializeMatrix(biases[0], 0);
	    initializeMatrix(weightsGradients[0], 0);
	    initializeMatrix(biasesGradients[0], 0);
	    initializeMatrix(outputs[0], 0);
	    initializeMatrix(activations[0], 0);
	    
	    Random rand = new Random();
		for (int i = 1; i < layerCount; i++) {
		    weights[i] = new Matrix(architecture[i-1], architecture[i]);
		    biases[i] = new Matrix(1, architecture[i]);
		    weightsGradients[i] = new Matrix(architecture[i-1], architecture[i]);
		    biasesGradients[i] = new Matrix(1, architecture[i]);
		    inputGradients[i] = new Matrix(architecture[i-1], architecture[i]);
		    outputs[i] = new Matrix(1, architecture[i]);
		    activations[i] = new Matrix(1, architecture[i]);
		    
		    initializeMatrixRand(weights[i], rand);
		    initializeMatrixRand(biases[i], rand);
		    initializeMatrix(weightsGradients[i], 0);
		    initializeMatrix(biasesGradients[i], 0);
		    initializeMatrix(outputs[i], 0);
		    initializeMatrix(activations[i], 0);
		}
		this.learningRate=0.5d;
		
		this.hiddenLayersAF="";
		this.outputLayerAF="";
		this.lossFunction="";
	}
	
	
	/**
	 * This function trains the neural network
	 * 
	 * @param trainingData: all columns dedicated to input apart from the last *nOutputs columns which will be used to store the expected output.
	 * @param nOutputs number of the coluns used for the expected outputs.
	 * @param batchSize the size of the single batches the training data will be split in, after which the weights and biases update. 1 for Stochastic gradient descent, 1<batchSize<trainingDataSize for mini batches, trainingDataSize for full batch 
	 * 
	 */
	public void train(Matrix trainingData, int nOutputs, int batchSize) {
		int trainCount=trainingData.getRows();
		
		// Loop over training examples
	    for (int i = 0; i < trainCount; ++i) {
	        // Forward pass
        	//System.out.println(i+" input[i] rows: " + trainingData.getSubMatrix(i, 0, 1, trainingData.getCols()-nOutputs).getRows() + " input[i] cols: " + trainingData.getSubMatrix(i, 0, 1, trainingData.getCols()-nOutputs).getCols());
	        forward(trainingData.getSubMatrix(i, 0, 1, trainingData.getCols()-nOutputs));
	        // Backwards pass
	        backPropagation(trainingData.getSubMatrix(i, trainingData.getCols()-nOutputs, 1, nOutputs));
	        if ((i + 1) % batchSize == 0 || i == trainingData.getCols() - 1) {  // Controlla se è multiplo di n
	    	    updateWeightsAndBiases(batchSize);

	        }
	    }
	}
	
	/**
	 * Forward propagation method, this function forwards the input through the neural network
	 * 
	 * @param input: 1d matrix of the input
	 * 
	 */
    public void forward(Matrix input) {
    	activations[0] = input;
    	outputs[0] = activations[0];
        for (int i = 1; i < layerCount; ++i) {
        	activations[i] = Matrix.multiply(activations[i-1], weights[i]);
        	activations[i].add(biases[i]);
            outputs[i]=activations[i];
            activations[i] = applyActivation(activations[i], i);
        }
    }
	
	/**
	 * This function is used to propagate the error of the output layer to all the hidden layers
	 * 
	 * @param expectedOutput the output that we expect from the neural network
	 * 
	 */
    public void backPropagation(Matrix expectedOutput) {
        for (int i = this.architecture.length - 1; i > 0; --i) {
        	for (int j = 0; j < architecture[i]; ++j) {                    
                double curNoutput = this.outputs[i].getElements()[0][j];
            	double curNactivatedOutput = this.activations[i].getElements()[0][j];
                double dActivationOnOutput=0;
                double delta;
                if (i == this.architecture.length - 1) { // Output layer
                	if (numOutputs > 1) { // Multi-output
                		delta = multipleOutputLossDerivative(curNactivatedOutput, expectedOutput.getElements()[0][j]); // because the af derivative and the loss derivative simplify each other only one calculation is needed
                    }else { // Single output
                		double dLoss_dY = lossDerivative(curNactivatedOutput, expectedOutput.getElements()[0][j]); // derivative of the loss function
                		delta = dLoss_dY * AFDerivative(curNoutput, this.outputLayerAF);// delta = dLoss_dY * derivative of the activation function with the non-activated output as input
                	}
                } else { // Hidden layers
                	dActivationOnOutput = AFDerivative(curNoutput, this.hiddenLayersAF); // derivative of the activation function with the non-activated output as input
                	double prevLayerGradientSum = 0;
                	for (int k = 0; k < architecture[i+1]; ++k) {
                        prevLayerGradientSum += inputGradients[i + 1].getElements()[j][k]; // considering the sum of the next layer input of the neuron considered
                    }
                	delta = prevLayerGradientSum * dActivationOnOutput; // applying chain rule on the partial derivatives calculated up to now. Delta is the same for every weight of a given neuron.
                }
                // update gradients.
                for (int k = 0; k < architecture[i - 1]; ++k) {
                    double weightGradient = delta * this.activations[i - 1].getElements()[0][k]; // calculating the gradient using the derivative of l(S(Z))
                    this.weightsGradients[i].getElements()[k][j] += weightGradient; // setting the weightGradient of the current neuron (swapped indexing)
                    
                    inputGradients[i].getElements()[k][j] = delta * weights[i].getElements()[k][j]; // storing the gradient of the input (swapped indexing)
                }
                // biasGradient = delta
                this.biasesGradients[i].getElements()[0][j] += delta; // Updated indexing (biases are now stored as (1 × Output Neurons))
            }
        }
    }

	/**
	 * This function is used to update weight and biases using gradients calculated during back propagation process.
	 * 
	 * @param trainCount the number of the training iterations
	 */
	private void updateWeightsAndBiases(int trainCount) {
	    for (int i = 1; i < this.architecture.length; ++i) {
	        for (int k = 0; k < this.architecture[i - 1]; ++k) { // Iterate over input neurons
	            for (int j = 0; j < this.architecture[i]; ++j) { // Iterate over output neurons
	                this.weightsGradients[i].getElements()[k][j] /= trainCount;
	                this.weights[i].getElements()[k][j] -= this.weightsGradients[i].getElements()[k][j] * learningRate;
	                this.weightsGradients[i].getElements()[k][j] = 0d;
	            } 
	        }
	        for (int j = 0; j < this.architecture[i]; ++j) {
	            this.biasesGradients[i].getElements()[0][j] /= trainCount;
	            this.biases[i].getElements()[0][j] -= this.biasesGradients[i].getElements()[0][j] * learningRate;
	            this.biasesGradients[i].getElements()[0][j] = 0d;
	        }
	    }
	}

	
	
	
	/**
	 * This function calculate the value of the loss for multiple outputs architectures
	 * 
	 * @param outputs the output of the nn
	 * @param expectedOutputs the output that we expect from the nn
	 * @return loss the loss value
	 */
	public double multipleOutputLoss(Matrix output, Matrix expectedOutput) {
		switch(this.lossFunction) {
		case "CCE":
			return CCEloss(output,expectedOutput);
		default:
            throw new IllegalArgumentException("Unsupported loss function: " + this.lossFunction);
		}
	}
	
	/**
	 * This function calculate the value of the derivative of the loss function for multiple outputs architectures
	 * 
	 * @param output the output of the nn
	 * @param expectedOutput the output that we expect from the nn
	 * @return loss the loss value
	 */
	public double multipleOutputLossDerivative(double output, double expectedOutput) {
		switch(this.lossFunction) {
		case "CCE":
			return CCElossDerivative(output,expectedOutput);
		default:
            throw new IllegalArgumentException("Unsupported loss function: " + this.lossFunction);
		}
	}
	
	/**
	 * This function calculate the value of the loss for single output architectures
	 * defaults to MSE loss function.
	 * 
	 * @param output the output of the nn
	 * @param expectedOutput the output that we expect from the nn
	 * @return loss the loss value
	 */
	public double loss(double output, double expectedOutput) {
		switch(this.lossFunction) {
		case "MSE":
			return MSEloss(output,expectedOutput);
		case "BCE":
			return BCEloss(output,expectedOutput);
		default:
			break;
		}
		return MSEloss(output,expectedOutput);
	}
	
	/**
	 * This function calculate the value of the derivative of the loss function for single output architectures
	 * 
	 * @param output the output of the nn
	 * @param expectedOutput the output that we expect from the nn
	 * @return loss the loss value
	 */
	public double lossDerivative(double output, double expectedOutput) {
		switch(this.lossFunction) {
		case "MSE":
			return MSElossDerivative(output,expectedOutput);
		case "BCE":
			return BCElossDerivative(output,expectedOutput);
		default:
			break;
		}
		return MSElossDerivative(output,expectedOutput);
	}
	
	
	/**
	 * This function applies the corresponding AFs to each layer.
	 * 
	 * @param matrix the output of a given layer to be activated
	 * @param iLayer the index of said layer
	 * @return activated[] the activated values
	 */
    private Matrix applyActivation(Matrix matrix, int iLayer) {
        Matrix activated = new Matrix(matrix.rows, matrix.cols);
        
        if((iLayer==layerCount-1) && (numOutputs>1)) { // try to apply non mutually exclusive multiple clases AFs first.
        	activated = multipleOutputActivationFunction(matrix);
        	if(activated!=null) {
        		return activated;
        	}
        }
        activated = new Matrix(matrix.rows, matrix.cols);
        // if they were not selected proceed with mutually exclusive AF.
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
            	if(iLayer==layerCount-1) {
            		activated.elements[i][j] = activationFunction(matrix.elements[i][j], this.outputLayerAF);
                }else {
                	activated.elements[i][j] = activationFunction(matrix.elements[i][j], this.hiddenLayersAF);
                }
            }
        }
        return activated;
    }
    
    private Matrix multipleOutputActivationFunction(Matrix input){
    	switch(this.outputLayerAF) {
		case "softmax":
			return softmax(input);
		default:
			break;
    	}
    	return null;
    }
    
    private Matrix MAFDerivative(Matrix input){
    	switch(this.outputLayerAF) {
		case "softmax":
			break;
		default:
			break;
    	}
    	return null;
    }
    
	private double activationFunction(double x, String af){
		switch(af) {
			case "sigmoid":
				return sigmoid(x);
			case "relu":
				return relu(x);
			default:
				break;
		}
		
		return x;
	}
    
    private double AFDerivative(double x, String af) {
    	
    	switch(af) {
		case "sigmoid":
			double sig = sigmoid(x);
	        return sig * (1.0 - sig);
		case "relu":
			return reluDerivative(x);
		default:
			break;
		}
		
		return 1;
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }
    
    private Matrix softmax(Matrix matrix) {
        Matrix result = new Matrix(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            double max = matrix.getElements()[i][0];
            for (int j = 1; j < matrix.cols; j++) {
                if (matrix.getElements()[i][j] > max) {
                    max = matrix.getElements()[i][j];
                }
            }
            double sum = 0.0;
            for (int j = 0; j < matrix.cols; j++) {
                result.getElements()[i][j] = Math.exp(matrix.getElements()[i][j] - max);
                sum += result.getElements()[i][j];
            }
            for (int j = 0; j < matrix.cols; j++) {
                result.getElements()[i][j] /= sum;
            }
        }
        return result;
    }
    
	
	
	public double CCEloss(Matrix predictions, Matrix labels) {
	    if (predictions.rows != labels.rows || predictions.cols != labels.cols) {
	        throw new IllegalArgumentException("Predictions and labels must have the same dimensions.");
	    }

	    double loss = 0.0;
	    for (int i = 0; i < predictions.rows; i++) {
	        for (int j = 0; j < predictions.cols; j++) {
	            double predicted = predictions.getElements()[i][j];
	            double expected = labels.getElements()[i][j];

	            // Ensure predicted values are valid probabilities
	            if (predicted <= 0 || predicted > 1) {
	                throw new IllegalArgumentException("Predictions must be probabilities (0 < p <= 1).");
	            }

	            // CCE formula: -sum(y * log(p))
	            loss += expected * Math.log(predicted + 1e-10); // Add epsilon to avoid log(0)
	        }
	    }

	    // Average the loss over all samples
	    return -loss / predictions.rows;
	}
	
	public double CCElossDerivative(double predicted, double expected) {
	    return predicted-expected;
	}
	
    
	public double BCEloss(double output, double expectedOutput) {
		// Clip output to avoid log(0)
	    double epsilon = 1e-9;  // Small value to prevent log(0)
	    output = Math.max(epsilon, Math.min(1 - epsilon, output));

	    return - (expectedOutput * Math.log(output) + (1 - expectedOutput) * Math.log(1 - output));
	}
	public double BCElossDerivative(double output, double expectedOutput) { 
		// Clip output to avoid division by zero
	    double epsilon = 1e-9;
	    output = Math.max(epsilon, Math.min(1 - epsilon, output));

	    return (output - expectedOutput) / (output * (1 - output));
	}

	public double MSEloss(double output, double expectedOutput) {
		double error=0d;
			error = (output - expectedOutput);
			error = error*error;
        return error;
	}

	public double MSElossDerivative(double output, double expectedOutput) {
		double error=0d;
			error = output - expectedOutput;
        return error;
	}
    
    
	/**
	 * Computes average loss (MSE) over a dataset.
	 * @param trainingData Matrix where each row is a training example, with inputs followed by expected outputs.
	 * @param nOutputs Number of output columns in trainingData.
	 * @return Average loss across all examples.
	 */
	public double computeAverageLoss(Matrix trainingData, int nOutputs) {
	    int numSamples = trainingData.getRows();
	    double totalLoss = 0.0;
	    
	    for (int i = 0; i < numSamples; i++) {
	        // Split input and expected output
	        Matrix input = trainingData.getSubMatrix(i, 0, 1, trainingData.getCols() - nOutputs);
	        Matrix expected = trainingData.getSubMatrix(i, trainingData.getCols() - nOutputs, 1, nOutputs);
	        
	        // Forward pass
	        forward(input);
	        Matrix prediction = activations[layerCount - 1];
	        
	        if(nOutputs>1) {
	        	totalLoss+=multipleOutputLoss(prediction, expected);
	        }else {
	        	// Calculate loss for this example
		        for (int j = 0; j < nOutputs; j++) {
		            totalLoss+=loss(prediction.getElements()[0][j], expected.getElements()[0][j]);
		        }
	        }    
	    }
	    return totalLoss / numSamples;
	}
	
	
	public double computeAccuracy(Matrix dataset, int nOutputs) {
		if(nOutputs>1) {
			return computeMultiClassAccuracy(dataset, nOutputs);
		}else {
			return computeSingleOutputAccuracy(dataset);
		}
	}
	
	public double computeMultiClassAccuracy(Matrix dataset, int nOutputs) {
	    int correct = 0;
	    for(int i=0; i<dataset.rows; i++) {
	        Matrix input = dataset.getSubMatrix(i, 0, 1, dataset.cols - nOutputs);
	        Matrix output = dataset.getSubMatrix(i, dataset.cols - nOutputs, 1, nOutputs);
	        
	        forward(input);
	        Matrix pred = activations[layerCount-1];
	        
	        int predClass = 0;
	        double maxVal = pred.getElements()[0][0];
	        for(int j=1; j<nOutputs; j++) {
	            if(pred.getElements()[0][j] > maxVal) {
	                maxVal = pred.getElements()[0][j];
	                predClass = j;
	            }
	        }
	        
	        int trueClass = 0;
	        for(int j=0; j<nOutputs; j++) {
	            if(output.getElements()[0][j] == 1.0) {
	                trueClass = j;
	                break;
	            }
	        }
	        
	        if(predClass == trueClass) correct++;
	    }
	    return (double)correct/dataset.rows*100;
	}
	
	public double computeSingleOutputAccuracy(Matrix dataset) {
	    int numSamples = dataset.getRows();
	    int correct = 0;
	    for (int i = 0; i < numSamples; i++) {
	        Matrix input = dataset.getSubMatrix(i, 0, 1, dataset.getCols() - 1);
	        Matrix expected = dataset.getSubMatrix(i, dataset.getCols() - 1, 1, 1);
	        forward(input);
	        double prediction = this.activations[layerCount - 1].getElements()[0][0];
	        int predictedLabel = (prediction >= 0.5) ? 1 : 0;
	        int trueLabel = (int) expected.getElements()[0][0];
	        if (predictedLabel == trueLabel) {
	            correct++;
	        }
	    }
	    return (double) correct / numSamples * 100; // Accuracy in percentage
	}
	
	private void initializeMatrixRand(Matrix matrix, Random rand) {
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                matrix.elements[i][j] = rand.nextGaussian(); // Random values from a normal distribution
            }
        }
    }
	private void initializeMatrix(Matrix matrix, double d) {
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                matrix.elements[i][j] = d; // Random values from a normal distribution
            }
        }
    }
	
	public void debugMatrixDimensions() {
	    for (int i = 0; i < layerCount; i++) {
	        System.out.println("Layer " + i);
	        System.out.println("Weights: " + weights[i].rows + " rows x " + weights[i].cols + " cols");
	        System.out.println("Biases: " + biases[i].rows + " rows x " + biases[i].cols + " cols");
	        System.out.println("Weights Gradients: " + weightsGradients[i].rows + " rows x " + weightsGradients[i].cols + " cols");
	        System.out.println("Biases Gradients: " + biasesGradients[i].rows + " rows x " + biasesGradients[i].cols + " cols");
	        System.out.println("Outputs: " + outputs[i].rows + " rows x " + outputs[i].cols + " cols");
	        System.out.println("Activations: " + activations[i].rows + " rows x " + activations[i].cols + " cols");
	        System.out.println();
	    }
	}
	
	
	
	/**
	 * 
	 * This method is used to save the state of the neural network
	 * 
	 * @return void
	 */
	public boolean saveState() {
		boolean saved = false;
		
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("savedNN.dat"));
			
			// writing the object to savedNN.dat and the closing the oos
			oos.writeObject(this);
			oos.close();
			
			// setting the saved value to true
			saved = true;
			
			// printing the completion of the save
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println(e.toString());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println(e.toString());
		}
		
		return saved;
	}
	
	/**
	 * 
	 * This function loads the state of the previous neural network
	 * 
	 * @return void
	 */
	public static NeuralNetwork loadState(){
		
		NeuralNetwork loadedNN = null;
		
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream("savedNN.dat"));
				
			loadedNN = (NeuralNetwork) ois.readObject();	// reading the serialize NN
			
			ois.close();	// closing the input stream
					
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		
		
		return loadedNN;
	}
	
	
	
	
	public String getHiddenLayersAF() {
		return hiddenLayersAF;
	}
	public void setHiddenLayersAF(String hiddenLayersAF) {
		this.hiddenLayersAF = hiddenLayersAF;
	}
	public String getOutputLayerAF() {
		return outputLayerAF;
	}
	public void setOutputLayerAF(String af) {
	    this.outputLayerAF = af;
	    if(af.equals("softmax")) {
	        this.lossFunction = "CCE"; // automatically set loss function to CCE
	    }
	}
	public String getLossFunction() {
		return lossFunction;
	}
	public void setLossFunction(String lossFunction) {
		this.lossFunction = lossFunction;
	}
	public double getLearning_rate() {
		return learningRate;
	}

	public void setLearning_rate(double learningRate) {
		this.learningRate = learningRate;
	}
}
