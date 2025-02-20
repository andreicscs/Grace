/*
 * 
 * 
 * neural network object
 * creates and manages a neural network.
 * 
 * 
 * */


/*
		WHAT TO CHANGE:
	Weights:
		The weights will be stored in an array of matrices:
		L_i = layers
		N_i = neurons
		W_i = weights
		
	Biases:
		The biases will be stored in an array of arrays
		
	Activations:
		Activations (activated neurons outputs) will be stored in an array of arrays.
		
	Outputs:
		Outputs(pre-activated neurons outputs, plain output or Z) will be stored in an array of arrays. 
		//used to apply the chain rule
		
	Architecture:
		The architecture will be declared by a vector, each column will represent a layer and its value will represent how many neurons that layer has.
		Ex:
			Int[] arch = {2,10,10,5,1};
			Represents the following architecture:
			2 neurons in the input layer, 10 neurons in the first hidden layer… 1 neuron in the output layer, for a total of 5 layers (arch.size).
	
	Arrays of arrays/matrices are used instead of directly using matrices or 3d matrices because 
	each layer can have different sizes, and the neuron's weights too. 
	So by using an array of matrices each layer can have a Dynamically allocated matrix 
	based on the architecture of the NN.
	
	By organizing weights and biases into arrays and matrices, it becomes easier to implement parallel computations. 
	Libraries like OpenMP for multithreading or CUDA for GPU acceleration can efficiently handle these operations on matrices.
	
	
	Matrices:
		N (first value) = columns
		M (second value) = rows
		
		first value of array of matrices initialized to n=1 m=architecture[0], 1 input means 1 weight, 1 bias... and there each row rapresents a neuron.
		other values initialized to architecture[i-1] architecture[i]
*/




// TO DO!!! not finished

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
	
	private double learningRate;
	//private double momentumFactor; // Represents how much of the momentum is retained ( to be implemented)

	
	// !TO DO check if initialization is correct
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
		this.learningRate=0.01d;
		
		// default to relu for hiudden layers and none for output layer can be changed using set...().
		this.hiddenLayersAF="rel";
		this.outputLayerAF="";
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
	 * This function trains the neural network
	 * 
	 * @param trainingData all columns dedicated to input apart from the last nOutputs columns which will be used to store the expected output.
	 * @param nOutputs number of the coluns used for the expected outputs
	 */
	public void train(Matrix trainingData, int nOutputs, int batchSize) {
		int trainCount=trainingData.getCols();
		
		// Loop over training examples
	    for (int i = 0; i < trainCount; ++i) {
	        // Forward pass
	        forward(trainingData.getSubMatrix(i, 0, 1, trainCount-nOutputs));
	        // Backwards pass
	        backPropagation(trainingData.getSubMatrix(i, trainCount-nOutputs, 1, nOutputs));
	        if ((i + 1) % batchSize == 0 || i == trainCount - 1) {  // Controlla se è multiplo di n
	    	    updateWeightsAndBiases(batchSize);

	        }
	    }
	}
	
	
	// TO DO non funge, controllare operazioni tra matrici.
	// Forward propagation method
    public void forward(Matrix input) {
    	activations[0] = input;
    	outputs[0] = activations[0];

        for (int i = 1; i < layerCount; i++) {
        	//System.out.println(i-1 + " Activations: " + activations[i-1].rows + " rows x " + activations[i-1].cols + " cols");
        	//System.out.println(i + " Weights: " + Matrix.transpose(weights[i]).rows + " rows x " + Matrix.transpose(weights[i]).cols + " cols");

        	activations[i] = Matrix.multiply(activations[i-1], weights[i]);
        	//System.out.println(i + " Activations: " + activations[i].rows + " rows x " + activations[i].cols + " cols");

        	activations[i].add(biases[i]);
            outputs[i]=activations[i];
            activations[i] = applyActivation(activations[i], i);
        }
        
    }
	
	/**
	 * This function is used to propagate the error of the output layer to all the hidden layers
	 * 
	 * @param expectedOutput the output that we expect from the neural network
	 */
    public void backPropagation(Matrix expectedOutput) {
        for (int i = this.architecture.length - 1; i > 0; --i) {
            for (int j = 0; j < architecture[i]; ++j) {                    
                double curNoutput = this.outputs[i].getElements()[0][j];
                double dActivationOnOutput = AFDerivative(curNoutput, this.outputLayerAF); // derivative of the activation function with the non-activated output as input
                
                double prevLayerGradientSum = 0;

                if (i == this.architecture.length - 1) { // Output layer
                    prevLayerGradientSum = lossDerivative(this.activations[i].getElements()[0][j], expectedOutput.getElements()[0][j]); // derivative of the loss function
                } else { // Hidden layers
                    for (int k = 0; k < architecture[i+1]; ++k) {
                        prevLayerGradientSum += inputGradients[i + 1].getElements()[j][k]; // considering the sum of the next layer input of the neuron considered
                    }
                }
                double delta = prevLayerGradientSum * dActivationOnOutput; // applying chain rule on the partial derivatives calculated up to now. Delta is the same for every weight of a given neuron.

                for (int k = 0; k < architecture[i - 1]; ++k) {
                    double weightGradient = delta * this.activations[i - 1].getElements()[0][k]; // calculating the gradient using the derivative of l(S(Z))
                    this.weightsGradients[i].getElements()[k][j] += weightGradient; // setting the weightGradient of the current neuron (swapped indexing)
                    
                    inputGradients[i].getElements()[k][j] = delta * weights[i].getElements()[k][j]; // storing the gradient of the input (swapped indexing)
                }

                // delta = biasGradient
                this.biasesGradients[i].getElements()[0][j] += delta; // Updated indexing (biases are now stored as (1 × Output Neurons))
            }
        }
    }

	
	
	
	/**
	 * This function is used to update weight and biases using each gradient calculated with the chain rule
	 * 
	 * @param trainCount the number of the train iterations
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
	 * This function calculate the value of the loss
	 * 
	 * @param output the output of the nn
	 * @param expectedOutput the output that we expect from the nn
	 * @return loss the loss value
	 */
	public double loss(double output, double expectedOutput) {
		double error=0d;
			error = (output - expectedOutput);
			error = Math.pow(error, 2);
        return error;
	}
	
	/**
	 * This function calculate the derivative of the loss function in the point x(weight)
	 * 
	 * @param x the point in x in the function loss
	 * @return the derivative of the loss(x)
	 */
	public double lossDerivative(double output, double expectedOutput) {
		double error=0d;
			error = output - expectedOutput;
        return error;
	}

	/**
	 * Computes average loss (MSE) over a dataset.
	 * @param trainingData Matrix where each row is a training example, 
	 *                     with inputs followed by expected outputs.
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
	        
	        // Calculate loss for this example
	        for (int j = 0; j < nOutputs; j++) {
	            double error = prediction.getElements()[0][j] - expected.getElements()[0][j];
	            totalLoss += error * error;
	        }
	    }
	    
	    // Average loss: total / (number of samples * number of outputs)
	    return totalLoss / (numSamples * nOutputs);
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
	 * This function load the state of the previous neural network
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

	
	/**
	 * 
	 * This function is used to make a trained neural network make calculated guesses on given inputs and debugging it
	 * 
	 * @param inputs The inputs that the neural network need to do the guessing on
	 * @param expectedOutputs The outputs that we expect from the neural network
	 * @return A list of all the calculated guess of the neural network
	 */
	/*
	public List<Double> nnGuessing(List<List<Double>> inputs, List<Double> expectedOutputs){
		List<Double> calculatedOutputGuess = null;
		if(NeuralNetwork.loadState() != null)
		{
			calculatedOutputGuess = new ArrayList<Double>();
			int inputsNumber = inputs.size();
			int wronGuess = 0;
			
			for (int k = 0; k < inputsNumber; ++k) {
		        // Forward pass to make the trained neural network guess the output
				calculatedOutputGuess.add(forward(inputs.get(k)).get(0));
		    }
			
			for(int k = 0; k < expectedOutputs.size(); k++) {
				if((Math.abs(calculatedOutputGuess.get(k)-expectedOutputs.get(k)) > 0.15)) {
					wronGuess++;
				}
			}
			
			for(int i = 0; i<calculatedOutputGuess.size(); i++) {
				System.out.print("\tExpected output: "+expectedOutputs.get(i).toString());
    	        System.out.print(" | Actual output: "+ calculatedOutputGuess.get(i).toString());
    	        System.out.println(" \tError: [ "+ Math.abs(expectedOutputs.get(i) - calculatedOutputGuess.get(i)) + " ]");
			}
			System.out.println(" \tThe percentage of error is: " + (double)wronGuess/expectedOutputs.size() * 100 + "%");
			
		} else
			System.out.println("Impossibile fare il guessing da una rete neurale non trainata");
		return calculatedOutputGuess;
	}
	*/
	/**
	 * 
	 * This function is used to make a trained neural network make calculated guesses on given inputs
	 * 
	 * @param inputs The inputs that the neural network need to do the guessing on
	 * @return A list of all the calculated guess of the neural network
	 */
	/**
	public List<List<Double>> nnGuessing(List<List<Double>> inputs){
		List<List<Double>> calculatedOutputGuess = null;
		if(NeuralNetwork.loadState() != null)
		{
			calculatedOutputGuess = new ArrayList<List<Double>>();
			int inputsNumber = inputs.size();
			
			for (int k = 0; k < inputsNumber; ++k) {
		        // Forward pass to make the trained neural network guess the output
				calculatedOutputGuess.add(forward(inputs.get(k)));
		    }
			
		} else
			System.out.println("Impossibile fare il guessing da una rete neurale non trainata");
		return calculatedOutputGuess;
	}
	*/
	
	
	
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
	public double getLearning_rate() {
		return learningRate;
	}

	public void setLearning_rate(double learningRate) {
		this.learningRate = learningRate;
	}
	
    
    private Matrix applyActivation(Matrix matrix, int iLayer) {
        Matrix activated = new Matrix(matrix.rows, matrix.cols);
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
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }
	private double activationFunction(double x, String af){
		switch(af) {
			case "sig":
				return sigmoid(x);
			case "rel":
				return relu(x);
			default:
				break;
		}
		
		return x;
	}
    
    private double AFDerivative(double x, String af) {
    	
    	switch(af) {
		case "sig":
			double sig = sigmoid(x);
	        return sig * (1.0 - sig);
		case "rel":
			return reluDerivative(x);
		default:
			break;
		}
		
		return 1;
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
	public void setOutputLayerAF(String outputLayerAF) {
		this.outputLayerAF = outputLayerAF;
	}
	
	
	
}
