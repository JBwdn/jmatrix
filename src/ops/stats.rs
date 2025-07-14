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
