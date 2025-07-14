use crate::generic_matrix::GenericMatrix;
use crate::ops::properties;
use std::ops::{Add, Mul};

pub fn matmul<T>(matrix: &GenericMatrix<T>, other: &GenericMatrix<T>) -> GenericMatrix<T>
where
    T: Clone + Default + Mul<Output = T> + Add<Output = T>,
{
    let (rows_a, cols_a) = properties::shape(matrix);
    let (rows_b, cols_b) = properties::shape(other);
    assert_eq!(
        cols_a, rows_b,
        "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
        rows_a, cols_a, rows_b, cols_b
    );

    let mut result = vec![vec![T::default(); cols_b]; rows_a];

    for (i, row) in result.iter_mut().enumerate().take(rows_a) {
        for (j, cell) in row.iter_mut().enumerate().take(cols_b) {
            let mut sum = T::default();
            for k in 0..cols_a {
                sum = sum + matrix.data[i][k].clone() * other.data[k][j].clone();
            }
            *cell = sum;
        }
    }
    GenericMatrix { data: result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::initialisers;

    #[test]
    fn test_matmul_basic_i32() {
        let matrix_a = initialisers::matrix_from_data(vec![vec![1, 2], vec![3, 4]]);
        let matrix_b = initialisers::matrix_from_data(vec![vec![5, 6], vec![7, 8]]);

        let result = matmul(&matrix_a, &matrix_b);

        // Expected result: [[19, 22], [43, 50]]
        assert_eq!(result.data, vec![vec![19, 22], vec![43, 50],]);
    }

    #[test]
    fn test_matmul_basic_f64() {
        let matrix_a = initialisers::matrix_from_data(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let matrix_b = initialisers::matrix_from_data(vec![vec![0.5, 1.5], vec![2.5, 3.5]]);

        let result = matmul(&matrix_a, &matrix_b);

        // Expected result: [[5.5, 8.5], [11.5, 18.5]]
        assert_eq!(result.data, vec![vec![5.5, 8.5], vec![11.5, 18.5],]);
    }

    #[test]
    fn test_matmul_identity_matrix() {
        let matrix = initialisers::matrix_from_data(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let identity =
            initialisers::matrix_from_data(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);

        let result = matmul(&matrix, &identity);

        // Multiplying by identity should return the original matrix
        assert_eq!(result.data, matrix.data);
    }

    #[test]
    fn test_matmul_non_square_matrices() {
        let matrix_a = initialisers::matrix_from_data(vec![vec![1, 2, 3], vec![4, 5, 6]]); // 2x3 matrix

        let matrix_b = initialisers::matrix_from_data(vec![vec![7, 8], vec![9, 10], vec![11, 12]]); // 3x2 matrix

        let result = matmul(&matrix_a, &matrix_b);

        // Expected result: 2x2 matrix
        // [[58, 64], [139, 154]]
        assert_eq!(result.data, vec![vec![58, 64], vec![139, 154],]);
    }

    #[test]
    fn test_matmul_single_element() {
        let matrix_a = initialisers::matrix_from_data(vec![vec![5]]);
        let matrix_b = initialisers::matrix_from_data(vec![vec![3]]);

        let result = matmul(&matrix_a, &matrix_b);

        assert_eq!(result.data, vec![vec![15]]);
    }

    #[test]
    fn test_matmul_with_zeros() {
        let matrix_a = initialisers::matrix_from_data(vec![vec![1, 0], vec![0, 1]]);
        let matrix_b = initialisers::matrix_from_data(vec![vec![0, 5], vec![3, 0]]);

        let result = matmul(&matrix_a, &matrix_b);

        assert_eq!(result.data, vec![vec![0, 5], vec![3, 0],]);
    }

    #[test]
    fn test_matmul_row_vector_times_column_vector() {
        let row_vector = initialisers::matrix_from_data(vec![vec![1, 2, 3]]); // 1x3
        let col_vector = initialisers::matrix_from_data(vec![vec![4], vec![5], vec![6]]); // 3x1

        let result = matmul(&row_vector, &col_vector);

        // Result should be 1x1 matrix with value 32 (1*4 + 2*5 + 3*6)
        assert_eq!(result.data, vec![vec![32]]);
    }

    #[test]
    fn test_matmul_column_vector_times_row_vector() {
        let col_vector = initialisers::matrix_from_data(vec![vec![1], vec![2], vec![3]]); // 3x1
        let row_vector = initialisers::matrix_from_data(vec![vec![4, 5, 6]]); // 1x3

        let result = matmul(&col_vector, &row_vector);

        // Result should be 3x3 matrix
        assert_eq!(
            result.data,
            vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18],]
        );
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions incompatible for multiplication")]
    fn test_matmul_incompatible_dimensions() {
        let matrix_a = initialisers::matrix_from_data(vec![vec![1, 2], vec![3, 4]]); // 2x2

        let matrix_b =
            initialisers::matrix_from_data(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]); // 3x3

        // This should panic because 2x2 * 3x3 is incompatible
        matmul(&matrix_a, &matrix_b);
    }

    #[test]
    fn test_matmul_rectangular_compatible() {
        let matrix_a = initialisers::matrix_from_data(vec![vec![1, 2, 3, 4]]); // 1x4

        let matrix_b = initialisers::matrix_from_data(vec![vec![1], vec![2], vec![3], vec![4]]); // 4x1

        let result = matmul(&matrix_a, &matrix_b);

        // Result should be 1x1 matrix with value 30 (1*1 + 2*2 + 3*3 + 4*4)
        assert_eq!(result.data, vec![vec![30]]);
    }

    #[test]
    fn test_matmul_associativity() {
        let a = initialisers::matrix_from_data(vec![vec![1, 2], vec![3, 4]]);
        let b = initialisers::matrix_from_data(vec![vec![5, 6], vec![7, 8]]);
        let c = initialisers::matrix_from_data(vec![vec![9, 10], vec![11, 12]]);

        // Test (A * B) * C = A * (B * C)
        let ab = matmul(&a, &b);
        let ab_c = matmul(&ab, &c);

        let bc = matmul(&b, &c);
        let a_bc = matmul(&a, &bc);

        assert_eq!(ab_c.data, a_bc.data);
    }
}
