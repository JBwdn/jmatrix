use crate::generic_matrix::GenericMatrix;
use crate::ops::properties;
use std::ops::{Add, Div, Mul, Sub};

pub fn add<T>(matrix: &GenericMatrix<T>, other: &GenericMatrix<T>) -> GenericMatrix<T>
where
    T: Clone + Add<Output = T>,
{
    assert_eq!(properties::shape(matrix), properties::shape(other));
    let mut result = Vec::new();

    for (i, row) in matrix.data.iter().enumerate() {
        let mut new_row = Vec::new();

        for (j, element) in row.iter().enumerate() {
            new_row.push(element.clone() + other.data[i][j].clone());
        }

        result.push(new_row);
    }

    GenericMatrix { data: result }
}

pub fn div<T>(matrix: &GenericMatrix<T>, other: &GenericMatrix<T>) -> GenericMatrix<T>
where
    T: Clone + Div<Output = T>,
{
    assert_eq!(properties::shape(matrix), properties::shape(other));
    let mut result = Vec::new();

    for (i, row) in matrix.data.iter().enumerate() {
        let mut new_row = Vec::new();

        for (j, element) in row.iter().enumerate() {
            new_row.push(element.clone() / other.data[i][j].clone());
        }

        result.push(new_row);
    }

    GenericMatrix { data: result }
}

pub fn sub<T>(matrix: &GenericMatrix<T>, other: &GenericMatrix<T>) -> GenericMatrix<T>
where
    T: Clone + Sub<Output = T>,
{
    assert_eq!(properties::shape(matrix), properties::shape(other));
    let mut result = Vec::new();

    for (i, row) in matrix.data.iter().enumerate() {
        let mut new_row = Vec::new();

        for (j, element) in row.iter().enumerate() {
            new_row.push(element.clone() - other.data[i][j].clone());
        }

        result.push(new_row);
    }

    GenericMatrix { data: result }
}

pub fn mul<T>(matrix: &GenericMatrix<T>, other: &GenericMatrix<T>) -> GenericMatrix<T>
where
    T: Clone + Mul<Output = T>,
{
    assert_eq!(properties::shape(matrix), properties::shape(other));
    let mut result = Vec::new();

    for (i, row) in matrix.data.iter().enumerate() {
        let mut new_row = Vec::new();

        for (j, element) in row.iter().enumerate() {
            new_row.push(element.clone() * other.data[i][j].clone());
        }

        result.push(new_row);
    }

    GenericMatrix { data: result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::initialisers;

    #[test]
    fn test_add_same_shape() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        let result = add(&matrix1, &matrix2);
        let expected = vec![
            vec![6.0, 8.0],
            vec![10.0, 12.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_add_single_element() {
        let matrix1 = initialisers::matrix_from_data(vec![vec![5.0]]);
        let matrix2 = initialisers::matrix_from_data(vec![vec![3.0]]);
        let result = add(&matrix1, &matrix2);
        assert_eq!(result.data, vec![vec![8.0]]);
    }

    #[test]
    fn test_add_with_zero() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let matrix2 = initialisers::zeros((2, 2));
        let result = add(&matrix1, &matrix2);
        assert_eq!(result.data, matrix1.data);
    }

    #[test]
    #[should_panic]
    fn test_add_different_shapes() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        add(&matrix1, &matrix2);
    }

    #[test]
    fn test_sub_same_shape() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = sub(&matrix1, &matrix2);
        let expected = vec![
            vec![4.0, 4.0],
            vec![4.0, 4.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_sub_with_self() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = sub(&matrix, &matrix);
        let expected = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    #[should_panic]
    fn test_sub_different_shapes() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![1.0],
            vec![2.0],
        ]);
        sub(&matrix1, &matrix2);
    }

    #[test]
    fn test_mul_same_shape() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![2.0, 3.0],
            vec![4.0, 5.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![6.0, 7.0],
            vec![8.0, 9.0],
        ]);
        let result = mul(&matrix1, &matrix2);
        let expected = vec![
            vec![12.0, 21.0],
            vec![32.0, 45.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_mul_with_ones() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![2.0, 3.0],
            vec![4.0, 5.0],
        ]);
        let matrix2 = initialisers::ones((2, 2));
        let result = mul(&matrix1, &matrix2);
        assert_eq!(result.data, matrix1.data);
    }

    #[test]
    fn test_mul_with_zero() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![2.0, 3.0],
            vec![4.0, 5.0],
        ]);
        let matrix2 = initialisers::zeros((2, 2));
        let result = mul(&matrix1, &matrix2);
        let expected = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    #[should_panic]
    fn test_mul_different_shapes() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        mul(&matrix1, &matrix2);
    }

    #[test]
    fn test_div_same_shape() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![12.0, 15.0],
            vec![20.0, 24.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![3.0, 5.0],
            vec![4.0, 6.0],
        ]);
        let result = div(&matrix1, &matrix2);
        let expected = vec![
            vec![4.0, 3.0],
            vec![5.0, 4.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_div_by_ones() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![2.0, 3.0],
            vec![4.0, 5.0],
        ]);
        let matrix2 = initialisers::ones((2, 2));
        let result = div(&matrix1, &matrix2);
        assert_eq!(result.data, matrix1.data);
    }

    #[test]
    fn test_div_single_element() {
        let matrix1 = initialisers::matrix_from_data(vec![vec![10.0]]);
        let matrix2 = initialisers::matrix_from_data(vec![vec![2.0]]);
        let result = div(&matrix1, &matrix2);
        assert_eq!(result.data, vec![vec![5.0]]);
    }

    #[test]
    #[should_panic]
    fn test_div_different_shapes() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
        ]);
        div(&matrix1, &matrix2);
    }

    #[test]
    fn test_operations_with_integers() {
        let matrix1 = initialisers::matrix_from_data(vec![
            vec![1, 2],
            vec![3, 4],
        ]);
        let matrix2 = initialisers::matrix_from_data(vec![
            vec![5, 6],
            vec![7, 8],
        ]);
        
        let add_result = add(&matrix1, &matrix2);
        let sub_result = sub(&matrix2, &matrix1);
        let mul_result = mul(&matrix1, &matrix2);
        
        assert_eq!(add_result.data, vec![vec![6, 8], vec![10, 12]]);
        assert_eq!(sub_result.data, vec![vec![4, 4], vec![4, 4]]);
        assert_eq!(mul_result.data, vec![vec![5, 12], vec![21, 32]]);
    }
}
