use crate::generic_matrix::GenericMatrix;
use crate::ops::properties;

pub fn broadcast<T: Clone>(matrix: &GenericMatrix<T>, other_shape: (usize, usize)) -> GenericMatrix<T> {
    let shape = properties::shape(matrix);
    let (rows, cols) = shape;
    let (target_rows, target_cols) = other_shape;

    // 1. they are equal (then leave matrix unchanged)
    if rows == target_rows && cols == target_cols {
        return matrix.clone();
    }

    if rows != target_rows && rows != 1 {
        panic!("Cannot broadcast: incompatible row dimensions {} and {}", rows, target_rows);
    }
    if cols != target_cols && cols != 1 {
        panic!("Cannot broadcast: incompatible column dimensions {} and {}", cols, target_cols);
    }

    // 2. matrix's size in that dim is 1 (repeat in that dim)
    let mut result = Vec::with_capacity(target_rows);

    for i in 0..target_rows {
        let mut row = Vec::with_capacity(target_cols);
        let source_row_idx = if rows == 1 { 0 } else { i };
        for j in 0..target_cols {
            let source_col_idx = if cols == 1 { 0 } else { j };
            row.push(matrix.data[source_row_idx][source_col_idx].clone());
        }
        result.push(row);
    }
    GenericMatrix { data: result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::initialisers;

    #[test]
    fn test_broadcast_same_shape() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = broadcast(&matrix, (2, 2));
        assert_eq!(result.data, matrix.data);
    }

    #[test]
    fn test_broadcast_single_element_to_matrix() {
        let matrix = initialisers::matrix_from_data(vec![vec![5.0]]);
        let result = broadcast(&matrix, (3, 2));
        let expected = vec![
            vec![5.0, 5.0],
            vec![5.0, 5.0],
            vec![5.0, 5.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_broadcast_row_vector() {
        let matrix = initialisers::matrix_from_data(vec![vec![1.0, 2.0, 3.0]]);
        let result = broadcast(&matrix, (4, 3));
        let expected = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_broadcast_column_vector() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
        ]);
        let result = broadcast(&matrix, (3, 4));
        let expected = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_broadcast_single_column_to_single_row() {
        let matrix = initialisers::matrix_from_data(vec![vec![42.0]]);
        let result = broadcast(&matrix, (1, 5));
        let expected = vec![vec![42.0, 42.0, 42.0, 42.0, 42.0]];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_broadcast_single_row_to_single_column() {
        let matrix = initialisers::matrix_from_data(vec![vec![7.0]]);
        let result = broadcast(&matrix, (3, 1));
        let expected = vec![vec![7.0], vec![7.0], vec![7.0]];
        assert_eq!(result.data, expected);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast: incompatible row dimensions")]
    fn test_broadcast_incompatible_rows() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]);
        broadcast(&matrix, (2, 2)); // Trying to broadcast 3x2 to 2x2
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast: incompatible column dimensions")]
    fn test_broadcast_incompatible_columns() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        broadcast(&matrix, (2, 2)); // Trying to broadcast 2x3 to 2x2
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast: incompatible row dimensions")]
    fn test_broadcast_incompatible_both_dimensions() {
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        broadcast(&matrix, (3, 3)); // Trying to broadcast 2x2 to 3x3 (both dims incompatible)
    }

    #[test]
    fn test_broadcast_with_integers() {
        let matrix = initialisers::matrix_from_data(vec![vec![1, 2]]);
        let result = broadcast(&matrix, (3, 2));
        let expected = vec![
            vec![1, 2],
            vec![1, 2],
            vec![1, 2],
        ];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_broadcast_larger_to_smaller_compatible() {
        // This should work when target is same as source
        let matrix = initialisers::matrix_from_data(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let result = broadcast(&matrix, (2, 2));
        assert_eq!(result.data, matrix.data);
    }
}
