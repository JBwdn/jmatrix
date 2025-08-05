use crate::generic_matrix::GenericMatrix;
use num_traits::{Float, FromPrimitive, One, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Sub};

use crate::ops::{broadcast, ewise, getters, initialisers, matmul, properties, stats};

impl<T> GenericMatrix<T>
where
    T: Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + One
        + Zero
        + AddAssign
        + FromPrimitive
        + Float,
{
    // Initialisers:
    pub fn new(data: Vec<Vec<T>>) -> Self {
        initialisers::matrix_from_data(data)
    }

    pub fn filled(shape: (usize, usize), value: T) -> Self {
        initialisers::filled_matrix(shape, value)
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        initialisers::zeros(shape)
    }

    pub fn ones(shape: (usize, usize)) -> Self {
        initialisers::ones(shape)
    }

    pub fn eye(size: usize) -> Self {
        initialisers::eye(size)
    }

    // Properties:
    pub fn shape(&self) -> (usize, usize) {
        properties::shape(self)
    }

    pub fn size(&self) -> usize {
        properties::size(self)
    }

    pub fn len(&self) -> usize {
        properties::len(self)
    }

    pub fn is_empty(&self) -> bool {
        properties::is_empty(self)
    }

    pub fn transpose(&self) -> Self {
        properties::transpose(self)
    }

    pub fn invert(&self) -> Result<Self, String> {
        properties::invert(self)
    }

    // Getters:
    pub fn get(&self, indices: (usize, usize)) -> T {
        getters::get(self, indices)
    }

    pub fn get_row(&self, index: usize) -> Vec<T> {
        getters::get_row(self, index)
    }

    pub fn get_col(&self, index: usize) -> Vec<T> {
        getters::get_col(self, index)
    }

    // Element-wise:
    pub fn ewise_add(&self, other: &Self) -> Self {
        ewise::add(self, other)
    }

    pub fn ewise_sub(&self, other: &Self) -> Self {
        ewise::sub(self, other)
    }

    pub fn ewise_mul(&self, other: &Self) -> Self {
        ewise::mul(self, other)
    }

    pub fn ewise_div(&self, other: &Self) -> Self {
        ewise::div(self, other)
    }

    // Matrix multiplication:
    pub fn matmul(&self, other: &Self) -> Self {
        matmul::matmul(self, other)
    }

    // Broadcasting:
    pub fn broadcast(&self, other_shape: (usize, usize)) -> Self {
        broadcast::broadcast(self, other_shape)
    }

    // Stats:
    pub fn mean(&self, axis: Option<usize>) -> Self {
        stats::mean(self, axis)
    }

    pub fn std(&self, axis: Option<usize>, ddof: Option<usize>) -> Self {
        stats::std(self, axis, ddof)
    }

    pub fn var(&self, axis: Option<usize>, ddof: Option<usize>) -> Self {
        stats::var(self, axis, ddof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_matrix_integration() {
        // Test that all methods work through the GenericMatrix interface
        let matrix = GenericMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        
        // Test properties
        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix.size(), 4);
        assert_eq!(matrix.len(), 2);
        assert!(!matrix.is_empty());
        
        // Test getters
        assert_eq!(matrix.get((0, 0)), 1.0);
        assert_eq!(matrix.get_row(0), vec![1.0, 2.0]);
        assert_eq!(matrix.get_col(0), vec![1.0, 3.0]);
        
        // Test transpose
        let transposed = matrix.transpose();
        assert_eq!(transposed.data, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    }

    #[test]
    fn test_matrix_operations_integration() {
        let matrix1 = GenericMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let matrix2 = GenericMatrix::new(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        
        // Test element-wise operations
        let add_result = matrix1.ewise_add(&matrix2);
        assert_eq!(add_result.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
        
        let sub_result = matrix2.ewise_sub(&matrix1);
        assert_eq!(sub_result.data, vec![vec![4.0, 4.0], vec![4.0, 4.0]]);
        
        let mul_result = matrix1.ewise_mul(&matrix2);
        assert_eq!(mul_result.data, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
        
        let div_result = matrix2.ewise_div(&matrix1);
        assert_eq!(div_result.data, vec![vec![5.0, 3.0], vec![7.0/3.0, 2.0]]);
    }

    #[test]
    fn test_matrix_matmul_integration() {
        let matrix1 = GenericMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let matrix2 = GenericMatrix::new(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        
        let result = matrix1.matmul(&matrix2);
        // [1,2] * [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        assert_eq!(result.data, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    }

    #[test]
    fn test_matrix_stats_integration() {
        let matrix = GenericMatrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        
        // Test mean
        let mean_axis_0 = matrix.mean(Some(0));
        assert_eq!(mean_axis_0.data, vec![vec![2.0], vec![5.0]]);
        
        let mean_axis_1 = matrix.mean(Some(1));
        assert_eq!(mean_axis_1.data, vec![vec![2.5, 3.5, 4.5]]);
        
        // Test that std and var work (just check they don't panic)
        let _std_result = matrix.std(Some(0), Some(1));
        let _var_result = matrix.var(Some(0), Some(1));
    }

    #[test]
    fn test_matrix_broadcast_integration() {
        let matrix = GenericMatrix::new(vec![vec![5.0]]);
        let result = matrix.broadcast((2, 3));
        let expected = vec![
            vec![5.0, 5.0, 5.0],
            vec![5.0, 5.0, 5.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_matrix_initialisers_integration() {
        // Test zeros
        let zeros = GenericMatrix::<f64>::zeros((2, 3));
        assert_eq!(zeros.data, vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
        
        // Test ones
        let ones = GenericMatrix::<f64>::ones((2, 2));
        assert_eq!(ones.data, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
        
        // Test eye
        let eye = GenericMatrix::<f64>::eye(3);
        let expected = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert_eq!(eye.data, expected);
        
        // Test filled
        let filled = GenericMatrix::filled((2, 2), 42.0);
        assert_eq!(filled.data, vec![vec![42.0, 42.0], vec![42.0, 42.0]]);
    }

    #[test]
    fn test_matrix_invert_integration() {
        // Test 2x2 matrix inversion
        let matrix = GenericMatrix::new(vec![
            vec![2.0, 1.0],
            vec![1.0, 1.0],
        ]);
        
        let inverted = matrix.invert().unwrap();
        let product = matrix.matmul(&inverted);
        let identity = GenericMatrix::<f64>::eye(2);
        
        // Check if product is close to identity (allowing for floating point errors)
        for i in 0..2 {
            for j in 0..2 {
                let diff = (product.data[i][j] - identity.data[i][j]).abs();
                assert!(diff < 1e-10, "Product should be identity matrix");
            }
        }
    }

    #[test]
    fn test_matrix_method_chaining() {
        // Test that methods can be chained together
        let matrix = GenericMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        
        let result = matrix
            .transpose()
            .ewise_add(&GenericMatrix::ones((2, 2)))
            .ewise_mul(&GenericMatrix::filled((2, 2), 2.0));
        
        // matrix.transpose() = [[1,3],[2,4]]
        // + ones = [[2,4],[3,5]]
        // * 2 = [[4,8],[6,10]]
        assert_eq!(result.data, vec![vec![4.0, 8.0], vec![6.0, 10.0]]);
    }

    #[test]
    fn test_matrix_with_different_types() {
        // Test with float matrices (since the impl requires Float trait)
        let float_matrix = GenericMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        
        assert_eq!(float_matrix.shape(), (2, 2));
        assert_eq!(float_matrix.get((1, 1)), 4.0);
        
        let ones = GenericMatrix::<f64>::ones((2, 2));
        let sum = float_matrix.ewise_add(&ones);
        assert_eq!(sum.data, vec![vec![2.0, 3.0], vec![4.0, 5.0]]);
    }
}
