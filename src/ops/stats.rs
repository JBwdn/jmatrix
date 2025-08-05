use crate::generic_matrix::GenericMatrix;
use crate::ops::{initialisers, properties};
use num_traits::{Float, FromPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Sub};

pub fn mean<T>(matrix: &GenericMatrix<T>, axis: Option<usize>) -> GenericMatrix<T>
where
    T: Clone + Zero + Add<Output = T> + Div<Output = T> + AddAssign + FromPrimitive,
{
    let axis = axis.unwrap_or(0); // Default to axis 0
    let shape = properties::shape(&matrix);

    match axis {
        0 => {
            let mut means = initialisers::zeros((shape.0, 1));
            for i in 0..shape.0 {
                let mut sum = T::zero();
                for j in 0..shape.1 {
                    sum += matrix.data[i][j].clone();
                }
                means.data[i][0] = sum / T::from_usize(shape.1).unwrap();
            }
            means
        }
        1 => {
            let mut means = initialisers::zeros((1, shape.1));
            for j in 0..shape.1 {
                let mut sum = T::zero();
                for i in 0..shape.0 {
                    sum += matrix.data[i][j].clone();
                }
                means.data[0][j] = sum / T::from_usize(shape.0).unwrap();
            }
            means
        }
        _ => panic!("Invalid axis: only 0 and 1 are supported for 2D matrices"),
    }
}

// Standard deviation function with optional axis and ddof (delta degrees of freedom)
pub fn std<T>(
    matrix: &GenericMatrix<T>,
    axis: Option<usize>,
    ddof: Option<usize>,
) -> GenericMatrix<T>
where
    T: Clone
        + Zero
        + Add<Output = T>
        + Div<Output = T>
        + AddAssign
        + FromPrimitive
        + Float
        + Sub<Output = T>
        + Mul<Output = T>,
{
    let axis = axis.unwrap_or(0); // Default to axis 0
    let ddof = ddof.unwrap_or(1); // Default to sample standard deviation (ddof=1)

    let shape = properties::shape(&matrix);

    match axis {
        0 => {
            // Standard deviation along rows (resulting in a column vector)
            let mut stds = initialisers::zeros((shape.0, 1));

            for i in 0..shape.0 {
                // Calculate mean first
                let mut sum = T::zero();
                for j in 0..shape.1 {
                    sum += matrix.data[i][j].clone();
                }
                let mean = sum / T::from_usize(shape.1).unwrap();

                // Calculate variance
                let mut variance_sum = T::zero();
                for j in 0..shape.1 {
                    let diff = matrix.data[i][j].clone() - mean.clone();
                    variance_sum += diff * diff;
                }

                // Apply degrees of freedom correction
                let n_minus_ddof = if shape.1 > ddof { shape.1 - ddof } else { 1 };
                let variance = variance_sum / T::from_usize(n_minus_ddof).unwrap();

                // Standard deviation is square root of variance
                stds.data[i][0] = variance.sqrt();
            }
            stds
        }
        1 => {
            // Standard deviation along columns (resulting in a row vector)
            let mut stds = initialisers::zeros((1, shape.1));

            for j in 0..shape.1 {
                // Calculate mean first
                let mut sum = T::zero();
                for i in 0..shape.0 {
                    sum += matrix.data[i][j].clone();
                }
                let mean = sum / T::from_usize(shape.0).unwrap();

                // Calculate variance
                let mut variance_sum = T::zero();
                for i in 0..shape.0 {
                    let diff = matrix.data[i][j].clone() - mean.clone();
                    variance_sum += diff * diff;
                }

                // Apply degrees of freedom correction
                let n_minus_ddof = if shape.0 > ddof { shape.0 - ddof } else { 1 };
                let variance = variance_sum / T::from_usize(n_minus_ddof).unwrap();

                // Standard deviation is square root of variance
                stds.data[0][j] = variance.sqrt();
            }
            stds
        }
        _ => panic!("Invalid axis: only 0 and 1 are supported for 2D matrices"),
    }
}

pub fn var<T>(
    matrix: &GenericMatrix<T>,
    axis: Option<usize>,
    ddof: Option<usize>,
) -> GenericMatrix<T>
where
    T: Clone
        + Zero
        + Add<Output = T>
        + Div<Output = T>
        + AddAssign
        + FromPrimitive
        + Float
        + Sub<Output = T>
        + Mul<Output = T>,
{
    let axis = axis.unwrap_or(0); // Default to axis 0
    let ddof = ddof.unwrap_or(1); // Default to sample variance (ddof=1)

    let shape = properties::shape(&matrix);

    match axis {
        0 => {
            // Variance along rows (resulting in a column vector)
            let mut vars = initialisers::zeros((shape.0, 1));

            for i in 0..shape.0 {
                // Calculate mean first
                let mut sum = T::zero();
                for j in 0..shape.1 {
                    sum += matrix.data[i][j].clone();
                }
                let mean = sum / T::from_usize(shape.1).unwrap();

                // Calculate variance
                let mut variance_sum = T::zero();
                for j in 0..shape.1 {
                    let diff = matrix.data[i][j].clone() - mean.clone();
                    variance_sum += diff * diff;
                }

                // Apply degrees of freedom correction
                let n_minus_ddof = if shape.1 > ddof { shape.1 - ddof } else { 1 };
                vars.data[i][0] = variance_sum / T::from_usize(n_minus_ddof).unwrap();
            }
            vars
        }
        1 => {
            // Variance along columns (resulting in a row vector)
            let mut vars = initialisers::zeros((1, shape.1));

            for j in 0..shape.1 {
                // Calculate mean first
                let mut sum = T::zero();
                for i in 0..shape.0 {
                    sum += matrix.data[i][j].clone();
                }
                let mean = sum / T::from_usize(shape.0).unwrap();

                // Calculate variance
                let mut variance_sum = T::zero();
                for i in 0..shape.0 {
                    let diff = matrix.data[i][j].clone() - mean.clone();
                    variance_sum += diff * diff;
                }

                // Apply degrees of freedom correction
                let n_minus_ddof = if shape.0 > ddof { shape.0 - ddof } else { 1 };
                vars.data[0][j] = variance_sum / T::from_usize(n_minus_ddof).unwrap();
            }
            vars
        }
        _ => panic!("Invalid axis: only 0 and 1 are supported for 2D matrices"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::initialisers;

    #[test]
    fn test_mean_axis_0() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let result = mean(&matrix, Some(0));
        let expected = vec![
            vec![2.0],  // (1+2+3)/3 = 2
            vec![5.0],  // (4+5+6)/3 = 5
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_mean_axis_1() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let result = mean(&matrix, Some(1));
        let expected = vec![
            vec![2.5, 3.5, 4.5],  // (1+4)/2, (2+5)/2, (3+6)/2
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_mean_default_axis() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = mean(&matrix, None); // Should default to axis 0
        let expected = vec![
            vec![1.5],  // (1+2)/2 = 1.5
            vec![3.5],  // (3+4)/2 = 3.5
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_mean_single_element() {
        let matrix = initialisers::matrix_from_data(vec![vec![5.0]]);
        let result = mean(&matrix, Some(0));
        assert_eq!(result.data, vec![vec![5.0]]);
        
        let result = mean(&matrix, Some(1));
        assert_eq!(result.data, vec![vec![5.0]]);
    }

    #[test]
    fn test_mean_single_row() {
        let matrix = initialisers::matrix_from_data(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        let result = mean(&matrix, Some(0));
        assert_eq!(result.data, vec![vec![2.5]]); // (1+2+3+4)/4 = 2.5
        
        let result = mean(&matrix, Some(1));
        assert_eq!(result.data, vec![vec![1.0, 2.0, 3.0, 4.0]]); // Each column averaged across 1 row
    }

    #[test]
    fn test_mean_single_column() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ]);
        let result = mean(&matrix, Some(0));
        let expected = vec![vec![1.0], vec![2.0], vec![3.0]]; // Each row averaged across 1 column
        assert_eq!(result.data, expected);
        
        let result = mean(&matrix, Some(1));
        assert_eq!(result.data, vec![vec![2.0]]); // (1+2+3)/3 = 2
    }

    #[test]
    #[should_panic(expected = "Invalid axis: only 0 and 1 are supported for 2D matrices")]
    fn test_mean_invalid_axis() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        mean(&matrix, Some(2));
    }

    #[test]
    fn test_std_axis_0() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = std(&matrix, Some(0), Some(1)); // Sample std (ddof=1)
        
        // Row 0: [1.0, 2.0], mean = 1.5, std = sqrt(((1-1.5)^2 + (2-1.5)^2) / 1) = sqrt(0.5) ≈ 0.707
        // Row 1: [3.0, 4.0], mean = 3.5, std = sqrt(((3-3.5)^2 + (4-3.5)^2) / 1) = sqrt(0.5) ≈ 0.707
        assert!((result.data[0][0] - 0.7071067811865476).abs() < 1e-10);
        assert!((result.data[1][0] - 0.7071067811865476).abs() < 1e-10);
    }

    #[test]
    fn test_std_axis_1() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 3.0],
            vec![2.0, 4.0],
        ]);
        let result = std(&matrix, Some(1), Some(1)); // Sample std (ddof=1)
        
        // Col 0: [1.0, 2.0], mean = 1.5, std = sqrt(((1-1.5)^2 + (2-1.5)^2) / 1) = sqrt(0.5) ≈ 0.707
        // Col 1: [3.0, 4.0], mean = 3.5, std = sqrt(((3-3.5)^2 + (4-3.5)^2) / 1) = sqrt(0.5) ≈ 0.707
        assert!((result.data[0][0] - 0.7071067811865476).abs() < 1e-10);
        assert!((result.data[0][1] - 0.7071067811865476).abs() < 1e-10);
    }

    #[test]
    fn test_std_ddof_0() {
        // Population standard deviation (ddof=0)
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = std(&matrix, Some(0), Some(0));
        
        // Row 0: [1.0, 2.0], mean = 1.5, std = sqrt(((1-1.5)^2 + (2-1.5)^2) / 2) = sqrt(0.25) = 0.5
        assert!((result.data[0][0] - 0.5).abs() < 1e-10);
        assert!((result.data[1][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_var_axis_0() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = var(&matrix, Some(0), Some(1)); // Sample variance (ddof=1)
        
        // Row 0: [1.0, 2.0], mean = 1.5, var = ((1-1.5)^2 + (2-1.5)^2) / 1 = 0.5
        assert!((result.data[0][0] - 0.5).abs() < 1e-10);
        assert!((result.data[1][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_var_axis_1() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 3.0],
            vec![2.0, 4.0],
        ]);
        let result = var(&matrix, Some(1), Some(1)); // Sample variance (ddof=1)
        
        // Col 0: [1.0, 2.0], mean = 1.5, var = ((1-1.5)^2 + (2-1.5)^2) / 1 = 0.5
        assert!((result.data[0][0] - 0.5).abs() < 1e-10);
        assert!((result.data[0][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_var_ddof_0() {
        // Population variance (ddof=0)
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = var(&matrix, Some(0), Some(0));
        
        // Row 0: [1.0, 2.0], mean = 1.5, var = ((1-1.5)^2 + (2-1.5)^2) / 2 = 0.25
        assert!((result.data[0][0] - 0.25).abs() < 1e-10);
        assert!((result.data[1][0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_stats_default_parameters() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        
        let mean_result = mean(&matrix, None); // Should default to axis 0
        let std_result = std(&matrix, None, None); // Should default to axis 0, ddof 1
        let var_result = var(&matrix, None, None); // Should default to axis 0, ddof 1
        
        // Verify shapes are correct
        assert_eq!(mean_result.data.len(), 2);
        assert_eq!(mean_result.data[0].len(), 1);
        assert_eq!(std_result.data.len(), 2);
        assert_eq!(std_result.data[0].len(), 1);
        assert_eq!(var_result.data.len(), 2);
        assert_eq!(var_result.data[0].len(), 1);
    }

    #[test]
    #[should_panic(expected = "Invalid axis: only 0 and 1 are supported for 2D matrices")]
    fn test_std_invalid_axis() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        std(&matrix, Some(2), Some(1));
    }

    #[test]
    #[should_panic(expected = "Invalid axis: only 0 and 1 are supported for 2D matrices")]
    fn test_var_invalid_axis() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        var(&matrix, Some(3), Some(1));
    }

    #[test]
    fn test_stats_single_element() {
        let matrix = initialisers::matrix_from_data(vec![vec![5.0]]);
        
        let mean_result = mean(&matrix, Some(0));
        let std_result = std(&matrix, Some(0), Some(1));
        let var_result = var(&matrix, Some(0), Some(1));
        
        assert_eq!(mean_result.data, vec![vec![5.0]]);
        // For single element, std and var should be 0 (or handled gracefully)
        assert_eq!(std_result.data, vec![vec![0.0]]);
        assert_eq!(var_result.data, vec![vec![0.0]]);
    }
}
