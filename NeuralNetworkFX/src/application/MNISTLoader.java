package application;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class MNISTLoader {

    public static Matrix loadMNIST(String filePath, int numSamples) {
        Matrix dataset = new Matrix(numSamples, 785); // 784 pixels + 1 label
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int row = 0;
            boolean isHeader = true; // Flag to skip the header row
            while ((line = br.readLine()) != null && row < numSamples) {
                if (isHeader) {
                    isHeader = false; // Skip the header row
                    continue;
                }
                String[] values = line.split(",");
                for (int col = 0; col < 785; col++) {
                    dataset.elements[row][col] = Double.parseDouble(values[col]);
                }
                row++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataset;
    }

    public static Matrix normalizeData(Matrix data) {
        // Normalize pixel values to [0, 1]
        for (int i = 0; i < data.rows; i++) {
            for (int j = 1; j < 785; j++) { // Skip the label column (column 0)
                data.elements[i][j] /= 255.0;
            }
        }
        return data;
    }

    public static Matrix oneHotEncodeLabels(Matrix data) {
        // Convert labels to one-hot encoded format
        Matrix labels = new Matrix(data.rows, 10);
        for (int i = 0; i < data.rows; i++) {
            int label = (int) data.elements[i][0];
            labels.elements[i][label] = 1.0;
        }
        return labels;
    }

    public static Matrix prepareDataset(Matrix data, Matrix labels) {
        // Combine inputs (pixel values) and one-hot encoded labels into a single matrix
        Matrix dataset = new Matrix(data.rows, 794); // 784 inputs + 10 outputs
        for (int i = 0; i < data.rows; i++) {
            // Copy pixel values (columns 1-784 of data)
            for (int j = 1; j < 785; j++) {
                dataset.elements[i][j - 1] = data.elements[i][j];
            }
            // Copy one-hot encoded labels (columns 785-794)
            for (int j = 0; j < 10; j++) {
                dataset.elements[i][784 + j] = labels.elements[i][j];
            }
        }
        return dataset;
    }
}