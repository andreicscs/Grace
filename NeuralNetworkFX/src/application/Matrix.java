package application;

import java.io.Serializable;

public class Matrix implements Serializable{
	private static final long serialVersionUID = 1L;
	
	int rows;
    int cols;
    double[][] elements;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.elements = new double[rows][cols];
    }

    public int getRows() {
        return rows;
    }

    public void setRows(int rows) {
        this.rows = rows;
    }

    public int getCols() {
        return cols;
    }

    public void setCols(int cols) {
        this.cols = cols;
    }

    public double[][] getElements() {
        return elements;
    }

    public void setElements(double[][] elements) {
        this.elements = elements;
    }
    
    /**
     * Sets the elements of the matrix at the specified row and column.
     * 
     * @param row The row index (0-based).
     * @param col The column index (0-based).
     * @param values The values to set (as a 1D array).
     */
    public void setElements(int row, int col, double[] values) {
        for (int i = 0; i < values.length; i++) {
            if (col + i < this.cols) { // Ensure column index is within bounds
                this.elements[row][col + i] = values[i];
            } else {
                throw new IllegalArgumentException("Column index out of bounds.");
            }
        }
    }
    public Matrix sumColumns() {
        // The result matrix will have 1 row and as many columns as the original matrix
        Matrix result = new Matrix(1, cols); 
        for (int j = 0; j < cols; j++) {
            double sum = 0;
            for (int i = 0; i < rows; i++) {
                sum += elements[i][j];
            }
            result.elements[0][j] = sum;
        }

        return result;
    }
    public static Matrix multiply(Matrix a, Matrix b) {
        if (a.cols != b.rows) throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
        Matrix result = new Matrix(a.rows, b.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.cols; j++) {
                for (int k = 0; k < a.cols; k++) {
                    result.elements[i][j] += a.elements[i][k] * b.elements[k][j];
                }
            }
        }
        return result;
    }

    public void add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Matrix dimensions must match for addition.");
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                this.elements[i][j] += other.elements[i][j];
            }
        }
    }

    public Matrix getSubMatrix(int startRow, int startCol, int numRows, int numCols) {
        if (startRow < 0 || startRow + numRows > rows || startCol < 0 || startCol + numCols > cols) {
            throw new IllegalArgumentException("Submatrix dimensions are out of bounds.");
        }
        
        Matrix subMatrix = new Matrix(numRows, numCols);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                subMatrix.elements[i][j] = this.elements[startRow + i][startCol + j];
            }
        }
        return subMatrix;
    }
    
    public static Matrix transpose(Matrix matrix) {
        int rows = matrix.getRows();
        int cols = matrix.getCols();
        Matrix transposed = new Matrix(cols, rows);
        
        // Transpose the matrix by swapping rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.getElements()[j][i] = matrix.getElements()[i][j];
            }
        }
        
        return transposed;
    }
    public Matrix transpose() {
        int rows = this.getRows();
        int cols = this.getCols();
        Matrix transposed = new Matrix(cols, rows);
        
        // Transpose the matrix by swapping rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.getElements()[j][i] = this.getElements()[i][j];
            }
        }
        
        return transposed;
    }
    public void printMatrix() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%8.3f ", elements[i][j]); // Format to 3 decimal places
            }
            System.out.println(); // New line after each row
        }
        System.out.println(); // Extra line for better readability
    }
    
    
    public static void scaleInPlace(Matrix matrix, double scalar) {
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                matrix.elements[i][j] *= scalar;
            }
        }
    }

    public static void subtractInPlace(Matrix a, Matrix b) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for subtraction.");
        }
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                a.elements[i][j] -= b.elements[i][j];
            }
        }
    }
    // Method to subtract another matrix from the current matrix
    public static Matrix subtract(Matrix a, Matrix b) {
        // Check if matrices have the same dimensions
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException("Matrices must have the same dimensions to subtract.");
        }
        Matrix result = new Matrix(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.elements[i][j] = a.elements[i][j] - b.elements[i][j];
            }
        }
        return result;
    }
    public static void addInPlace(Matrix a, Matrix b) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition.");
        }
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                a.elements[i][j] += b.elements[i][j];
            }
        }
    }
    // Method for element-wise multiplication of two matrices
    public static Matrix multiplyElementWise(Matrix a, Matrix b) {
        // Check if matrices have the same dimensions
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for element-wise multiplication.");
        }
        Matrix result = new Matrix(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.elements[i][j] = a.elements[i][j] * b.elements[i][j];
            }
        }
        return result;
    }
    public Matrix scale(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.elements[i][j] = this.elements[i][j] * scalar;
            }
        }
        return result;
    }

    public static void fill(Matrix matrix, double value) {
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                matrix.elements[i][j] = value;
            }
        }
    }
    
}
