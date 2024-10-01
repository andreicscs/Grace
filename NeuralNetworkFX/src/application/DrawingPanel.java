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
	            	NeuralNetwork scervelo = new NeuralNetwork();
	            	ArrayList<List<Double>> TrainIn = new ArrayList<>();
	            	ArrayList<List<Double>> TrainOut = new ArrayList<>();
	        		ArrayList<List<Double>> GuessIn = new ArrayList<>();
	        		ArrayList<Double> GuessOut = new ArrayList<Double>();
	            	File nnData = new File("savedNN.dat");
	            	int i;

	        		
	            	if(!toTrain && nnData.exists())
	            		scervelo = NeuralNetwork.loadState();
	        		
	        		
	            	CsvReader dataset = DataReader.getCSV("dataset.csv");
	            	ArrayList<Cancer> data = new ArrayList<Cancer>();
		
					// for debugging purpose only
					//System.out.println(dataset.toString());
					
					try {
						
						dataset.readHeaders();	// reading the headers of the csv
						
						while (dataset.readRecord()) {	// populating the cancers dataset
							data.add(new Cancer(dataset.get("id"), dataset.get("diagnosis"), Float.parseFloat(dataset.get("radius_mean")), Float.parseFloat(dataset.get("texture_mean")), 
									Float.parseFloat(dataset.get("perimeter_mean")), Float.parseFloat(dataset.get("area_mean")), Float.parseFloat(dataset.get("smoothness_mean")),
									Float.parseFloat(dataset.get("compactness_mean")), Float.parseFloat(dataset.get("concavity_mean")), Float.parseFloat(dataset.get("concave points_mean")),
									Float.parseFloat(dataset.get("symmetry_mean")), Float.parseFloat(dataset.get("fractal_dimension_mean")), Float.parseFloat(dataset.get("radius_se")),
									Float.parseFloat(dataset.get("texture_se")), Float.parseFloat(dataset.get("perimeter_se")), Float.parseFloat(dataset.get("area_se")), Float.parseFloat(dataset.get("smoothness_se")),
									Float.parseFloat(dataset.get("compactness_se")), Float.parseFloat(dataset.get("concavity_se")), Float.parseFloat(dataset.get("concave points_se")),
									Float.parseFloat(dataset.get("symmetry_se")), Float.parseFloat(dataset.get("fractal_dimension_se")), Float.parseFloat(dataset.get("radius_worst")),
									Float.parseFloat(dataset.get("texture_worst")), Float.parseFloat(dataset.get("perimeter_worst")), Float.parseFloat(dataset.get("area_worst")),
									Float.parseFloat(dataset.get("smoothness_worst")), Float.parseFloat(dataset.get("compactness_worst")), Float.parseFloat(dataset.get("concavity_worst")),
									Float.parseFloat(dataset.get("concave points_worst")), Float.parseFloat(dataset.get("symmetry_worst")), Float.parseFloat(dataset.get("fractal_dimension_worst"))));
						}
					} catch (NumberFormatException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
					dataset.close();
	        		
	        		// setting up all the training data for the nn to use
	        		for(i = 0; i < data.size()-50; i++) {
	        			double diagnosis;
	        			
	        			if("M".equals(data.get(i).getDiagnosis()))
	        				diagnosis = 0;
	        			else
	        				diagnosis = 1;
	        			
	        			TrainOut.add(List.of(diagnosis));
	        			TrainIn.add(data.get(i).getAllNormalizedData());
	        			
	        			// for debugging purpose only
	        			//System.out.println("first training data: " + TrainIn.get(i).toString());
	        			//System.out.println("number of inputs: " + TrainIn.get(i).size());
	        		}
	        		
	        		// setting up the guess data
	        		for(i++; i < data.size(); i++) {
	        			double diagnosis;
	        			
	        			if("M".equals(data.get(i).getDiagnosis()))
	        				diagnosis = 0;
	        			else
	        				diagnosis = 1;
	        			
	        			GuessOut.add(diagnosis);
	        			GuessIn.add(data.get(i).getAllNormalizedData());
	        			
	        			// for debugging purpose only
	        			//System.out.println("first training data: " + TrainIn.get(i).toString());
	        			//System.out.println("number of inputs: " + TrainIn.get(i).size());
	        		}
	        		
	            	
	        		

	        		if(toTrain) {
	        			double startTime = System.currentTimeMillis();
	        			double endTime;
	        			double elapsedTime;
	        			for(i=0; i<3000; ++i) {
		        			scervelo.train(TrainIn, TrainOut);
		        			// DEBUG
		        			if(i%100==0) {
		        				endTime = System.currentTimeMillis();
		        				elapsedTime = endTime - startTime;
		        				System.out.println("Iteration " + i + /*", Cost: " + scervelo.lossAverage(TrainIn, TrainOut) + */",time: " + elapsedTime);
		        				startTime = endTime;
		        			}
	        				
		        		}
	        		} else {
	        		}
	        	
	        		
	        		/*
	        		 * 
	        		 * DEBUGGING
	        		 * 
	        		*/
	            }
           });
	}
	 
	public void start(boolean toTrain) {
		this.toTrain = toTrain;
		thread1.start();
	}

	
}
