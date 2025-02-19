package application;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
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
	            	scervelo.debugMatrixDimensions();
	            	
	            	Matrix trainingData = new Matrix(3,3);
	            	
	            	
	            	File nnData = new File("savedNN.dat");
	            	
	        		if(toTrain) {
	        			double startTime = System.currentTimeMillis();
	        			double endTime;
	        			double elapsedTime;
	        			
	        			for(int i=0; i<3000; ++i) {
		        			scervelo.train(trainingData, 1, 1);
		        			// DEBUG
		        			if(i%100==0) {
		        				endTime = System.currentTimeMillis();
		        				elapsedTime = endTime - startTime;
		        				System.out.println("Iteration " + i + /*", Cost: " + scervelo.lossAverage(TrainIn, TrainOut) + */",time: " + elapsedTime);
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
