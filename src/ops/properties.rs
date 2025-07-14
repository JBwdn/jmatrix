use crate::generic_matrix::GenericMatrix;
use std::ops::{Add, Div, Mul, Sub};

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
        + Sub<Output = T>,
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
        augmented[i][i + n] = T::default() + T::default() + T::default() + T::default();
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
