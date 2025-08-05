use pyo3::prelude::*;

pub mod generic_matrix;
use crate::generic_matrix::GenericMatrix;

#[macro_use]
pub mod create_matrix_interface;

pub mod generic_matrix_methods;

pub mod ops;

pub mod vector;

create_matrix_interface!(FloatMatrix, f64);

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_comprehensive_matrix_operations() {
        // Create a comprehensive test of matrix operations
        let matrix1 = GenericMatrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        
        let matrix2 = GenericMatrix::new(vec![
            vec![9.0, 8.0, 7.0],
            vec![6.0, 5.0, 4.0],
            vec![3.0, 2.0, 1.0],
        ]);
        
        // Test various operations
        let sum = matrix1.ewise_add(&matrix2);
        let diff = matrix1.ewise_sub(&matrix2);
        let product_ewise = matrix1.ewise_mul(&matrix2);
        let quotient = matrix1.ewise_div(&matrix2);
        let matmul_result = matrix1.matmul(&matrix2);
        
        // Verify results
        assert_eq!(sum.data[0], vec![10.0, 10.0, 10.0]);
        assert_eq!(diff.data[0], vec![-8.0, -6.0, -4.0]);
        assert_eq!(product_ewise.data[0], vec![9.0, 16.0, 21.0]);
        assert_eq!(quotient.data[0], vec![1.0/9.0, 2.0/8.0, 3.0/7.0]);
        
        // Matrix multiplication result for first row:
        // [1,2,3] * [[9,8,7],[6,5,4],[3,2,1]] = [1*9+2*6+3*3, 1*8+2*5+3*2, 1*7+2*4+3*1] = [30, 24, 18]
        assert_eq!(matmul_result.data[0], vec![30.0, 24.0, 18.0]);
    }

    #[test]
    fn test_floating_point_precision() {
        // Test operations with small numbers to check precision
        let matrix1 = GenericMatrix::new(vec![
            vec![1e-10, 2e-10],
            vec![3e-10, 4e-10],
        ]);
        
        let matrix2 = GenericMatrix::new(vec![
            vec![1e-10, 1e-10],
            vec![1e-10, 1e-10],
        ]);
        
        let sum = matrix1.ewise_add(&matrix2);
        assert!((sum.data[0][0] - 2e-10_f64).abs() < 1e-15);
        
        let product = matrix1.ewise_mul(&matrix2);
        assert!((product.data[0][0] - 1e-20_f64).abs() < 1e-25);
    }

    #[test]
    fn test_operations_with_infinity() {
        let matrix1 = GenericMatrix::new(vec![vec![1.0, f64::INFINITY]]);
        let matrix2 = GenericMatrix::new(vec![vec![2.0, 1.0]]);
        
        let sum = matrix1.ewise_add(&matrix2);
        assert_eq!(sum.data[0][0], 3.0);
        assert!(sum.data[0][1].is_infinite());
        
        let product = matrix1.ewise_mul(&matrix2);
        assert_eq!(product.data[0][0], 2.0);
        assert!(product.data[0][1].is_infinite());
    }

    #[test]
    fn test_operations_with_nan() {
        let matrix1 = GenericMatrix::new(vec![vec![1.0, f64::NAN]]);
        let matrix2 = GenericMatrix::new(vec![vec![2.0, 1.0]]);
        
        let sum = matrix1.ewise_add(&matrix2);
        assert_eq!(sum.data[0][0], 3.0);
        assert!(sum.data[0][1].is_nan());
    }

    #[test]
    fn test_large_matrix_operations() {
        // Test with a reasonably large matrix to ensure performance is acceptable
        let size = 10;
        let mut data1 = vec![];
        let mut data2 = vec![];
        
        for i in 0..size {
            let mut row1 = vec![];
            let mut row2 = vec![];
            for j in 0..size {
                row1.push((i * size + j) as f64);
                row2.push(((i * size + j) + 1) as f64);
            }
            data1.push(row1);
            data2.push(row2);
        }
        
        let matrix1 = GenericMatrix::new(data1);
        let matrix2 = GenericMatrix::new(data2);
        
        // Test that operations complete without issues
        let _sum = matrix1.ewise_add(&matrix2);
        let _product = matrix1.matmul(&matrix2);
        let _transposed = matrix1.transpose();
        let _mean = matrix1.mean(Some(0));
    }

    #[test]
    fn test_chain_of_operations() {
        // Test a complex chain of operations
        let matrix = GenericMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        
        let result = matrix
            .ewise_add(&GenericMatrix::ones((2, 2)))
            .ewise_mul(&GenericMatrix::filled((2, 2), 2.0))
            .transpose()
            .ewise_sub(&GenericMatrix::filled((2, 2), 1.0));
        
        // [[1,2],[3,4]] + [[1,1],[1,1]] = [[2,3],[4,5]]
        // * [[2,2],[2,2]] = [[4,6],[8,10]]
        // transpose = [[4,8],[6,10]]
        // - [[1,1],[1,1]] = [[3,7],[5,9]]
        assert_eq!(result.data, vec![vec![3.0, 7.0], vec![5.0, 9.0]]);
    }

    #[test]
    fn test_matrix_inversion_correctness() {
        // Test matrix inversion with a known invertible matrix
        let matrix = GenericMatrix::new(vec![
            vec![4.0, 7.0],
            vec![2.0, 6.0],
        ]);
        
        let inverted = matrix.invert().unwrap();
        let product = matrix.matmul(&inverted);
        let identity = GenericMatrix::<f64>::eye(2);
        
        // Check that A * A^(-1) ≈ I
        for i in 0..2 {
            for j in 0..2 {
                let diff = (product.data[i][j] - identity.data[i][j]).abs();
                assert!(diff < 1e-10, "Matrix inversion failed: element ({},{}) has error {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_statistical_operations_correctness() {
        // Test statistical operations with known values
        let matrix = GenericMatrix::new(vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
        ]);
        
        // Test mean along axis 0 (row means)
        let mean_axis_0 = matrix.mean(Some(0));
        assert_eq!(mean_axis_0.data[0][0], 3.0); // (1+2+3+4+5)/5 = 3
        assert_eq!(mean_axis_0.data[1][0], 6.0); // (2+4+6+8+10)/5 = 6
        
        // Test mean along axis 1 (column means)
        let mean_axis_1 = matrix.mean(Some(1));
        assert_eq!(mean_axis_1.data[0][0], 1.5);  // (1+2)/2 = 1.5
        assert_eq!(mean_axis_1.data[0][1], 3.0);  // (2+4)/2 = 3.0
        assert_eq!(mean_axis_1.data[0][2], 4.5);  // (3+6)/2 = 4.5
        
        // Test variance and standard deviation
        let var_result = matrix.var(Some(0), Some(0)); // Population variance
        let std_result = matrix.std(Some(0), Some(0)); // Population std dev
        
        // For the first row [1,2,3,4,5], mean=3, variance = ((1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2)/5 = (4+1+0+1+4)/5 = 2
        assert!((var_result.data[0][0] - 2.0_f64).abs() < 1e-10);
        assert!((std_result.data[0][0] - 2.0_f64.sqrt()).abs() < 1e-10);
    }
}

#[pymodule]
#[pyo3(name = "_lib")]
fn my_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FloatMatrix>()?;
    Ok(())
}
