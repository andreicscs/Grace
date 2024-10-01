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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class NeuralNetwork implements Serializable{

	private static final long serialVersionUID = 972856922459233840L;
	
	private final int[] architecture;
    private final Matrix[] weights;
    private final Matrix[] biases;
    private final Matrix[] outputs;
    private final Matrix[] activations;
    private final int layerCount;

	
	private double learningRate;
	//private double momentumFactor; // Represents how much of the momentum is retained ( to be implemented)

	
	// !TO DO check if initialization is correct
	public NeuralNetwork(int[] architecture) {
		super();
		this.architecture = architecture;
		this.layerCount = architecture.length;
		this.weights = new Matrix[layerCount - 1];
		this.biases = new Matrix[layerCount - 1];
		this.activations = new Matrix[layerCount - 1];
		this.outputs = new Matrix[layerCount - 1];
		
		
		weights[0] = new Matrix(architecture[0], architecture[0]);
	    biases[0] = new Matrix(1, architecture[0]);
	    outputs[0] = new Matrix(1, architecture[0]);
	    activations[0] = new Matrix(1, architecture[0]);
	    initializeMatrix(weights[0], 1);
	    initializeMatrix(biases[0], 0);
	    initializeMatrix(outputs[0], 0);
	    initializeMatrix(activations[0], 0);
	    
	    Random rand = new Random();
		for (int i = 1; i < layerCount - 1; i++) {
		    weights[i] = new Matrix(architecture[i], architecture[i]);
		    biases[i] = new Matrix(1, architecture[i]);
		    outputs[i] = new Matrix(1, architecture[i]);
		    activations[i] = new Matrix(1, architecture[i]);
		    
		    initializeMatrixRand(weights[i], rand);
		    initializeMatrixRand(biases[i], rand);
		    initializeMatrix(outputs[i], 0);
		    initializeMatrix(activations[i], 0);
		}
		this.learningRate=0.5d;
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
	public double getLearning_rate() {
		return learningRate;
	}

	public void setLearning_rate(double learningRate) {
		this.learningRate = learningRate;
	}
	

	
	
	// !TO DO check if forward is correct
	// Forward propagation method
    public void forward(Matrix input) {
    	activations[0] = input;
        for (int i = 0; i < layerCount - 1; i++) {
        	activations[i+1] = Matrix.multiply(activations[i], weights[i]);
            activations[i+1].add(biases[i]);
            outputs[i+1]=activations[i+1];
            activations[i+1] = applyActivation(activations[i+1]);
        }
    }
    
    private Matrix applyActivation(Matrix matrix) {
        Matrix activated = new Matrix(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                activated.elements[i][j] = sigmoid(matrix.elements[i][j]);
            }
        }
        return activated;
    }
    
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
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
			error = 2*(output - expectedOutput);
        return error;
	}
	
	
	
	
	// !TO DO implement backpropagation
	
	
	/**
	 * This function is used to propagate the error of the output layer to all the hidden layers
	 * 
	 * @param expectedOutput the output that we expect from the neural network
	 */
	/*
	public void backPropagation(List<Double> expectedOutput) {
		List<List<Double>> curLayerInGradients = new ArrayList<>();	// to store the current layer's gradients of the input which will be used in the next layer to apply the chain rule
		for(int i = this.getLayers().size() - 1; i>0; --i) {
	        List<Neuron> currentLayerNeurons = this.getLayers().get(i);
	        List<Neuron> previousLayerNeurons = this.getLayers().get(i - 1);
	        int neuronCount = currentLayerNeurons.size();
	        int prevNeuronCount = previousLayerNeurons.size();

			// Output layer
			if(i == this.getLayers().size()-1) {	
				for(int j = 0; j<neuronCount;++j) {
					Neuron currentNeuron=this.getLayers().get(i).get(j);
	                double curNoutput = currentNeuron.getOutput();
					double activatedCurNoutput = currentNeuron.activate(curNoutput);
					double dLoss = lossDerivative(activatedCurNoutput,expectedOutput.get(j)); // derivative of the loss function
					double dActivationOnOutput = currentNeuron.AFDerivative(curNoutput);	// derivative of the activation function with the non activated output as input
					double delta = dLoss*dActivationOnOutput;	// chain rule on the partial derivatives calculated up to now. Delta is the same for every weight of a given neuron.
	                List<Double> weightGradients = new ArrayList<>(Collections.nCopies(prevNeuronCount, 0.0));
					for(int k = 0; k < prevNeuronCount; k++){
						Neuron previousNeuron = previousLayerNeurons.get(k);	// taking the previous neuron that gives the weight the input
						double weightGradient = delta * previousNeuron.activate(previousNeuron.getOutput());	// calculating the gradient using the derivative of l(S(Z))	
						currentNeuron.setWeightGradient(k, currentNeuron.getWeightGradient(k) + weightGradient);	// setting the weightGradient of the current neuron
	                    weightGradients.set(k, weightGradients.get(k) + delta * currentNeuron.getWeight(k));
					}
	                curLayerInGradients.add(weightGradients);
					//	delta = biasGradient
					currentNeuron.setBiasGradient(currentNeuron.getBiasGradient() + delta);
				}	
			}
			else{ // Hidden layers
	            List<List<Double>> prevLayerInGradients = new ArrayList<>(curLayerInGradients);	// to store the previous layer's gradients of the input
	            curLayerInGradients.clear();

				for(int j = 0; j<neuronCount;++j) {
					Neuron currentNeuron=this.getLayers().get(i).get(j);
					double dActivationOnOutput = currentNeuron.AFDerivative(currentNeuron.getOutput());	// derivative of the activation function with the non activated output as input
					double prevLayerGradientSum = 0;
					for(int k=0; k< prevLayerInGradients.size(); ++k) {
						prevLayerGradientSum += prevLayerInGradients.get(k).get(j);	// considering the sum of the next layer input of the neuron considered
					}
					double delta = prevLayerGradientSum*dActivationOnOutput;	// chain rule on the partial derivatives calculated up to now. Delta is the same for every weight of a given neuron.
	                List<Double> weightGradients = new ArrayList<>(Collections.nCopies(prevNeuronCount, 0.0));
					for(int k = 0; k < prevNeuronCount; k++){
						Neuron previousNeuron = previousLayerNeurons.get(k);	// taking the previous neuron that gives the weight the input
						double weightGradient = delta * previousNeuron.activate(previousNeuron.getOutput());	// calculating the gradient using the derivative of l(S(Z))	
						currentNeuron.setWeightGradient(k, currentNeuron.getWeightGradient(k) + weightGradient);
	                    weightGradients.set(k, weightGradients.get(k) + delta * currentNeuron.getWeight(k));
					}
	                curLayerInGradients.add(weightGradients);
	                // delta = biasGradient
					currentNeuron.setBiasGradient(currentNeuron.getBiasGradient() + delta);
				}
			}
		}
	}
	
	
	
	// !TO DO rewrite train method using matrix as input/expected output.
	/**
	 * This function trains the neural network
	 * batch gradient descent
	 * 
	 * @param trainingData the inputs of the inputs layer
	 * @param outTrainingData the expected output
	 */
	public void train(List<List<Double>> trainingData, List<List<Double>> outTrainingData) {
		int trainCount=trainingData.size();
		// Loop over training examples
	    for (int i = 0; i < trainCount; ++i) {
	        // Forward pass
	        forward(trainingData.get(i));
	        // Backword pass
	        //backPropagation(outTrainingData.get(i));
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
}
