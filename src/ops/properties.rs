use crate::generic_matrix::GenericMatrix;
use std::ops::{Add, Div, Mul, Sub};
use num_traits::{One, Zero};

pub fn shape<T>(matrix: &GenericMatrix<T>) -> (usize, usize) {
    (matrix.data.len(), matrix.data[0].len())
}

pub fn size<T>(matrix: &GenericMatrix<T>) -> usize {
    matrix.data.len() * matrix.data[0].len()
}

pub fn len<T>(matrix: &GenericMatrix<T>) -> usize {
    matrix.data.len()
}

pub fn is_empty<T>(matrix: &GenericMatrix<T>) -> bool {
    matrix.data.is_empty()
}

pub fn get<T: Clone>(matrix: &GenericMatrix<T>, indices: (usize, usize)) -> T {
    matrix.data[indices.0][indices.1].clone()
}

pub fn transpose<T: Clone>(matrix: &GenericMatrix<T>) -> GenericMatrix<T> {
    let mut transposed = vec![Vec::with_capacity(matrix.data.len()); matrix.data[0].len()];
    for row in &matrix.data {
        for (j, val) in row.iter().enumerate() {
            transposed[j].push(val.clone());
        }
    }
    GenericMatrix { data: transposed }
}

pub fn invert<T>(matrix: &GenericMatrix<T>) -> Result<GenericMatrix<T>, String>
where
    T: Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + One
        + Zero,
{
    let (rows, cols) = shape(matrix);
    if rows != cols {
        return Err("Matrix must be square to invert".to_string());
    }

    let n = rows;

    // Create augmented matrix [A|I]
    let mut augmented = vec![vec![T::default(); 2 * n]; n];

    // Fill the left side with the original matrix
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = matrix.data[i][j].clone();
        }
    }
    // Fill the right side with identity matrix
    for i in 0..n {
        augmented[i][i + n] = T::one();
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut pivot_row = i;
        for k in i + 1..n {
            // This is simplified - in practice we'd need proper comparison
            if augmented[k][i] != T::default() {
                pivot_row = k;
                break;
            }
        }

        // Swap rows if needed
        if pivot_row != i {
            augmented.swap(i, pivot_row);
        }

        // Check for singular matrix
        if augmented[i][i] == T::default() {
            return Err("Matrix is singular and cannot be inverted".to_string());
        }

        // Scale pivot row
        let pivot = augmented[i][i].clone();
        for j in 0..2 * n {
            augmented[i][j] = augmented[i][j].clone() / pivot.clone();
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = augmented[k][i].clone();
                for j in 0..2 * n {
                    augmented[k][j] =
                        augmented[k][j].clone() - factor.clone() * augmented[i][j].clone();
                }
            }
        }
    }

    // Extract inverse matrix from right side
    let mut inverse_data = vec![vec![T::default(); n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse_data[i][j] = augmented[i][j + n].clone();
        }
    }

    Ok(GenericMatrix { data: inverse_data })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::initialisers;

    #[test]
    fn test_shape() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        assert_eq!(shape(&matrix), (2, 3));
    }

    #[test]
    fn test_shape_single_element() {
        let matrix = initialisers::matrix_from_data(vec![vec![42.0]]);
        assert_eq!(shape(&matrix), (1, 1));
    }

    #[test]
    fn test_shape_single_row() {
        let matrix = initialisers::matrix_from_data(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        assert_eq!(shape(&matrix), (1, 4));
    }

    #[test]
    fn test_shape_single_column() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ]);
        assert_eq!(shape(&matrix), (3, 1));
    }

    #[test]
    fn test_size() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        assert_eq!(size(&matrix), 6);
    }

    #[test]
    fn test_size_single_element() {
        let matrix = initialisers::matrix_from_data(vec![vec![42.0]]);
        assert_eq!(size(&matrix), 1);
    }

    #[test]
    fn test_len() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        assert_eq!(len(&matrix), 2); // Number of rows
    }

    #[test]
    fn test_len_single_row() {
        let matrix = initialisers::matrix_from_data(vec![vec![1.0, 2.0, 3.0]]);
        assert_eq!(len(&matrix), 1);
    }

    #[test]
    fn test_is_empty_false() {
        let matrix = initialisers::matrix_from_data(vec![vec![1.0]]);
        assert!(!is_empty(&matrix));
    }

    #[test]
    fn test_is_empty_true() {
        let matrix: GenericMatrix<f64> = GenericMatrix { data: vec![] };
        assert!(is_empty(&matrix));
    }

    #[test]
    fn test_get_valid_indices() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        assert_eq!(get(&matrix, (0, 0)), 1.0);
        assert_eq!(get(&matrix, (0, 2)), 3.0);
        assert_eq!(get(&matrix, (1, 1)), 5.0);
        assert_eq!(get(&matrix, (1, 2)), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_get_invalid_row() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        get(&matrix, (2, 0)); // Invalid row index
    }

    #[test]
    #[should_panic]
    fn test_get_invalid_col() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        get(&matrix, (0, 2)); // Invalid column index
    }

    #[test]
    fn test_transpose_square() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let transposed = transpose(&matrix);
        let expected = vec![
            vec![1.0, 3.0],
            vec![2.0, 4.0],
        ];
        assert_eq!(transposed.data, expected);
    }

    #[test]
    fn test_transpose_rectangular() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let transposed = transpose(&matrix);
        let expected = vec![
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
        ];
        assert_eq!(transposed.data, expected);
    }

    #[test]
    fn test_transpose_single_element() {
        let matrix = initialisers::matrix_from_data(vec![vec![42.0]]);
        let transposed = transpose(&matrix);
        assert_eq!(transposed.data, vec![vec![42.0]]);
    }

    #[test]
    fn test_transpose_single_row() {
        let matrix = initialisers::matrix_from_data(vec![vec![1.0, 2.0, 3.0]]);
        let transposed = transpose(&matrix);
        let expected = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(transposed.data, expected);
    }

    #[test]
    fn test_transpose_single_column() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ]);
        let transposed = transpose(&matrix);
        let expected = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(transposed.data, expected);
    }

    #[test]
    fn test_invert_2x2_identity() {
        let matrix: GenericMatrix<f64> = initialisers::eye(2);
        let inverted = invert(&matrix).unwrap();
        // Identity matrix should be its own inverse
        assert_eq!(inverted.data, matrix.data);
    }

    #[test]
    fn test_invert_2x2_simple() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let inverted = invert(&matrix).unwrap();
        
        // Verify A * A^-1 = I by manual calculation
        // Expected inverse of [[1,2],[3,4]] is [[-2, 1], [1.5, -0.5]]
        let product = crate::ops::matmul::matmul(&matrix, &inverted);
        let identity: GenericMatrix<f64> = initialisers::eye(2);
        
        // Check if product is close to identity (allowing for floating point errors)
        for i in 0..2 {
            for j in 0..2 {
                let diff: f64 = (product.data[i][j] - identity.data[i][j]).abs();
                assert!(diff < 1e-10, "Product should be identity matrix");
            }
        }
    }

    #[test]
    fn test_invert_non_square_matrix() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let result = invert(&matrix);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Matrix must be square to invert");
    }

    #[test]
    fn test_invert_singular_matrix() {
        // Create a singular matrix (rows are linearly dependent)
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0], // Second row is 2 * first row
        ]);
        let result = invert(&matrix);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Matrix is singular and cannot be inverted");
    }
}
