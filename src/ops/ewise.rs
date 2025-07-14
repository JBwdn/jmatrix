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
