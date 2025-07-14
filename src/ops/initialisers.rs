use crate::generic_matrix::GenericMatrix;
use num_traits::{One, Zero};

pub fn matrix_from_data<T>(data: Vec<Vec<T>>) -> GenericMatrix<T> {
    assert!(
        data.iter().all(|row| row.len() == data[0].len()),
        "All rows must have the same length"
    );
    GenericMatrix { data }
}

pub fn filled_matrix<T: Clone>(shape: (usize, usize), value: T) -> GenericMatrix<T> {
    let data = vec![vec![value; shape.1]; shape.0];
    matrix_from_data(data)
}

pub fn ones<T: Clone + One>(shape: (usize, usize)) -> GenericMatrix<T> {
    filled_matrix(shape, T::one())
}

pub fn zeros<T: Clone + Zero>(shape: (usize, usize)) -> GenericMatrix<T> {
    filled_matrix(shape, T::zero())
}

pub fn eye<T: Clone + Zero + One>(size: usize) -> GenericMatrix<T> {
    let mut data = vec![vec![T::zero(); size]; size];
    for (i, row) in data.iter_mut().enumerate().take(size) {
        row[i] = T::one()
    }
    GenericMatrix { data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_from_data() {
        let data = vec![vec![1, 2], vec![3, 4]];
        let mat = matrix_from_data(data.clone());
        assert_eq!(mat.data, data);
    }

    #[test]
    #[should_panic(expected = "All rows must have the same length")]
    fn test_matrix_from_data_invalid() {
        let data = vec![vec![1, 2], vec![3]];
        matrix_from_data(data);
    }

    #[test]
    fn test_filled_matrix() {
        let mat = filled_matrix((2, 3), 7);
        assert_eq!(mat.data, vec![vec![7, 7, 7], vec![7, 7, 7]]);
    }

    #[test]
    fn test_ones() {
        let mat = ones::<i32>((2, 2));
        assert_eq!(mat.data, vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn test_zeros() {
        let mat = zeros::<i32>((2, 2));
        assert_eq!(mat.data, vec![vec![0, 0], vec![0, 0]]);
    }

    #[test]
    fn test_eye() {
        let mat = eye::<i32>(3);
        assert_eq!(mat.data, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
    }
}
