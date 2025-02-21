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
	            	
	            	int[] architecture= {2,4,1};
	            	
	            	NeuralNetwork scervelo = new NeuralNetwork(architecture);
	            	//scervelo.debugMatrixDimensions();
	            	
	            	Matrix dataset = new Matrix(4, 3);
	                dataset.setElements(new double[][] {
	                	{0,0,0},
	                	{0,1,1},
	                	{1,0,1},
	                	{1,1,0}
	                });
	            	
	            	scervelo.setLearning_rate(0.1);
	            	scervelo.setHiddenLayersAF("relu");
	            	scervelo.setOutputLayerAF("sigmoid");
	            	int nOutputs=1;
	            	
	            	
	            	File nnData = new File("savedNN.dat");
	            	
	        		if(toTrain) {
	        			double startTime = System.currentTimeMillis();
	        			double endTime;
	        			double elapsedTime;
	        			
	        			for(int i=0; i<10000000; ++i) {
		        			scervelo.train(dataset, nOutputs, 64);
		        			// DEBUG
		        			if (i % 1000 == 0) {
		        				endTime = System.currentTimeMillis();
		        				elapsedTime = endTime - startTime;
		        				startTime = endTime;
		        		        double loss = scervelo.computeAverageLoss(dataset, nOutputs);
		        		        double accuracy = scervelo.computeAccuracy(dataset, nOutputs);
		        		        System.out.println("Iteration " + i + ",\t Cost: " + loss + ",\t Accuracy: " + accuracy +"%" + ",\t time (ms): " + elapsedTime);
		        			}
		        		}
	        		}else if(nnData.exists()) {
	        			scervelo = NeuralNetwork.loadState();
	        		}
	        		
	            }
           });
	}
	
	public void start(boolean toTrain) {
		this.toTrain = toTrain;
		thread1.start();
	}

	
}
