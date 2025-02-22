package application;

import java.io.File;
import java.io.IOException;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.StackPane;


public class DrawingPanel extends StackPane{
    private static Canvas canvas;
	private static GraphicsContext g2d;
	private Thread thread1;
	boolean toTrain;
	
	public DrawingPanel() throws IOException {
		canvas = new Canvas();
		
		getChildren().add(canvas);
		canvas.widthProperty().bind(widthProperty());
		canvas.heightProperty().bind(heightProperty());
		g2d = canvas.getGraphicsContext2D();
		g2d.setLineWidth(2.0);

		thread1 = new Thread(new Runnable() {
	            @Override
				public void run() {
	            	// Load and preprocess data
	                String trainFilePath = "C:\\Users\\termi\\git\\Grace\\NeuralNetworkFX\\src\\mnist_train.csv";
	                String testFilePath = "C:\\Users\\termi\\git\\Grace\\NeuralNetworkFX\\src\\mnist_test.csv";
	                int numTrainSamples = 60000; // MNIST training set size
	                int numTestSamples = 10000;  // MNIST test set size

	                // Load and preprocess training data
	                Matrix trainData = MNISTLoader.loadMNIST(trainFilePath, numTrainSamples);
	                trainData = MNISTLoader.normalizeData(trainData);
	                Matrix trainLabels = MNISTLoader.oneHotEncodeLabels(trainData);
	                Matrix trainDataset = MNISTLoader.prepareDataset(trainData, trainLabels);

	                // Load and preprocess test data
	                Matrix testData = MNISTLoader.loadMNIST(testFilePath, numTestSamples);
	                testData = MNISTLoader.normalizeData(testData);
	                Matrix testLabels = MNISTLoader.oneHotEncodeLabels(testData);
	                Matrix testDataset = MNISTLoader.prepareDataset(testData, testLabels);

	                // Now you can use trainDataset and testDataset with your neural network
	                // trainDataset: first 784 columns = inputs, last 10 columns = one-hot labels
	                // testDataset: first 784 columns = inputs, last 10 columns = one-hot labels

	                // Define network architecture
	                int[] architecture = {784, 128, 64, 10};
	                NeuralNetwork nn = new NeuralNetwork(architecture);
	                nn.setHiddenLayersAF("relu");
	                nn.setOutputLayerAF("softmax");
	                nn.setLearning_rate(0.1);

	                // Train the network
	                int nOutputs=10;
	                int batchSize = 32;
	                int epochs = 30;

	                for (int epoch = 0; epoch < epochs; epoch++) {
	                    System.out.println("Epoch " + (epoch + 1));
	                    
	                    long startTime = System.nanoTime(); // Start timing
	                    
	                    nn.train(trainDataset, nOutputs, batchSize);
	                    double trainLoss = nn.computeAverageLoss(trainDataset, nOutputs);
	                    double trainAccuracy = nn.computeAccuracy(trainDataset, nOutputs);
	                    System.out.println("Training Loss: " + trainLoss + ", Accuracy: " + trainAccuracy + "%");

	                    double testLoss = nn.computeAverageLoss(testDataset, nOutputs);
	                    double testAccuracy = nn.computeAccuracy(testDataset, nOutputs);
	                    System.out.println("Test Loss: " + testLoss + ", Accuracy: " + testAccuracy + "%");
	                    
	                    long endTime = System.nanoTime(); // End timing
	                    double elapsedTime = (endTime - startTime) / 1e9; // Convert nanoseconds to seconds
	                    System.out.println("Epoch Time: " + elapsedTime + " seconds\n");
	                }

	                // Save the trained model
	                nn.saveState();
	        		
	            }
           });
	}
	
	public void start(boolean toTrain) {
		this.toTrain = toTrain;
		thread1.start();
	}

	
}
