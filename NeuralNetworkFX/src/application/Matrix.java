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

    // Additional methods for filling, printing, etc.
}
