package application;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import javafx.application.Platform;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import com.csvreader.CsvReader;

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
	            	
	            	int[] architecture= {2,4,2,1};
	            	
	            	NeuralNetwork scervelo = new NeuralNetwork(architecture);
	            	//scervelo.debugMatrixDimensions();
	            	
	            	// Generate 1000 points
	            	Matrix dataset = new Matrix(1000, 3); // Columns: x, y, label
	            	Random rand = new Random();
	            	for (int i = 0; i < 1000; i++) {
	            	    double x = rand.nextDouble() * 2 - 1; // x in [-1, 1]
	            	    double y = rand.nextDouble() * 2 - 1; // y in [-1, 1]
	            	    double label = (x * x + y * y <= 0.7 * 0.7) ? 1 : 0; // Inside circle?
	            	    dataset.setElements(i, 0, new double[]{x, y, label});
	            	}
	            	
	            	
	            	File nnData = new File("savedNN.dat");
	            	
	        		if(toTrain) {
	        			double startTime = System.currentTimeMillis();
	        			double endTime;
	        			double elapsedTime;
	        			
	        			for(int i=0; i<10000; ++i) {
		        			scervelo.train(dataset, 1, 1);
		        			// DEBUG
		        			if(i%100==0) {
		        				endTime = System.currentTimeMillis();
		        				elapsedTime = endTime - startTime;
		        				System.out.println("Iteration " + i + ", Cost: " + scervelo.computeAverageLoss(dataset, 1) + ",time: " + elapsedTime);
		        				startTime = endTime;
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
