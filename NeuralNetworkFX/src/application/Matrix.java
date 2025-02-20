package application;

public class Matrix {
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

    
}
