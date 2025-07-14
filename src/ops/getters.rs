use crate::generic_matrix::GenericMatrix;

pub fn get<T: Clone>(matrix: &GenericMatrix<T>, indices: (usize, usize)) -> T {
    matrix.data[indices.0][indices.1].clone()
}

pub fn get_row<T: Clone>(matrix: &GenericMatrix<T>, index: usize) -> Vec<T> {
    matrix.data[index].clone()
}

pub fn get_col<T: Clone>(matrix: &GenericMatrix<T>, index: usize) -> Vec<T> {
    matrix.data.iter().map(|row| row[index].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic_matrix::GenericMatrix;

    // Helper function to create a test matrix
    fn create_test_matrix() -> GenericMatrix<i32> {
        GenericMatrix {
            data: vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]],
        }
    }

    fn create_string_matrix() -> GenericMatrix<String> {
        GenericMatrix {
            data: vec![
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec!["d".to_string(), "e".to_string(), "f".to_string()],
            ],
        }
    }

    fn create_single_element_matrix() -> GenericMatrix<i32> {
        GenericMatrix {
            data: vec![vec![42]],
        }
    }

    #[test]
    fn test_get_valid_indices() {
        let matrix = create_test_matrix();

        assert_eq!(get(&matrix, (0, 0)), 1);
        assert_eq!(get(&matrix, (0, 1)), 2);
        assert_eq!(get(&matrix, (0, 2)), 3);
        assert_eq!(get(&matrix, (1, 0)), 4);
        assert_eq!(get(&matrix, (1, 1)), 5);
        assert_eq!(get(&matrix, (1, 2)), 6);
        assert_eq!(get(&matrix, (2, 0)), 7);
        assert_eq!(get(&matrix, (2, 1)), 8);
        assert_eq!(get(&matrix, (2, 2)), 9);
    }

    #[test]
    fn test_get_single_element() {
        let matrix = create_single_element_matrix();
        assert_eq!(get(&matrix, (0, 0)), 42);
    }

    #[test]
    fn test_get_string_matrix() {
        let matrix = create_string_matrix();
        assert_eq!(get(&matrix, (0, 0)), "a");
        assert_eq!(get(&matrix, (1, 2)), "f");
    }

    #[test]
    #[should_panic]
    fn test_get_invalid_row_index() {
        let matrix = create_test_matrix();
        get(&matrix, (3, 0)); // Row index out of bounds
    }

    #[test]
    #[should_panic]
    fn test_get_invalid_col_index() {
        let matrix = create_test_matrix();
        get(&matrix, (0, 3)); // Column index out of bounds
    }

    #[test]
    #[should_panic]
    fn test_get_both_indices_invalid() {
        let matrix = create_test_matrix();
        get(&matrix, (5, 5)); // Both indices out of bounds
    }

    #[test]
    fn test_get_row_valid_indices() {
        let matrix = create_test_matrix();

        assert_eq!(get_row(&matrix, 0), vec![1, 2, 3]);
        assert_eq!(get_row(&matrix, 1), vec![4, 5, 6]);
        assert_eq!(get_row(&matrix, 2), vec![7, 8, 9]);
    }

    #[test]
    fn test_get_row_single_element() {
        let matrix = create_single_element_matrix();
        assert_eq!(get_row(&matrix, 0), vec![42]);
    }

    #[test]
    fn test_get_row_string_matrix() {
        let matrix = create_string_matrix();
        assert_eq!(get_row(&matrix, 0), vec!["a", "b", "c"]);
        assert_eq!(get_row(&matrix, 1), vec!["d", "e", "f"]);
    }

    #[test]
    #[should_panic]
    fn test_get_row_invalid_index() {
        let matrix = create_test_matrix();
        get_row(&matrix, 3); // Row index out of bounds
    }

    #[test]
    fn test_get_col_valid_indices() {
        let matrix = create_test_matrix();

        assert_eq!(get_col(&matrix, 0), vec![1, 4, 7]);
        assert_eq!(get_col(&matrix, 1), vec![2, 5, 8]);
        assert_eq!(get_col(&matrix, 2), vec![3, 6, 9]);
    }

    #[test]
    fn test_get_col_single_element() {
        let matrix = create_single_element_matrix();
        assert_eq!(get_col(&matrix, 0), vec![42]);
    }

    #[test]
    fn test_get_col_string_matrix() {
        let matrix = create_string_matrix();
        assert_eq!(get_col(&matrix, 0), vec!["a", "d"]);
        assert_eq!(get_col(&matrix, 1), vec!["b", "e"]);
        assert_eq!(get_col(&matrix, 2), vec!["c", "f"]);
    }

    #[test]
    #[should_panic]
    fn test_get_col_invalid_index() {
        let matrix = create_test_matrix();
        get_col(&matrix, 3); // Column index out of bounds
    }

    #[test]
    fn test_rectangular_matrix() {
        let matrix = GenericMatrix {
            data: vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]],
        };

        // Test get function
        assert_eq!(get(&matrix, (0, 3)), 4);
        assert_eq!(get(&matrix, (1, 0)), 5);

        // Test get_row function
        assert_eq!(get_row(&matrix, 0), vec![1, 2, 3, 4]);
        assert_eq!(get_row(&matrix, 1), vec![5, 6, 7, 8]);

        // Test get_col function
        assert_eq!(get_col(&matrix, 0), vec![1, 5]);
        assert_eq!(get_col(&matrix, 1), vec![2, 6]);
        assert_eq!(get_col(&matrix, 2), vec![3, 7]);
        assert_eq!(get_col(&matrix, 3), vec![4, 8]);
    }

    #[test]
    fn test_clone_independence() {
        let matrix = create_string_matrix();
        let mut retrieved = get(&matrix, (0, 0));
        retrieved.push_str("_modified");

        // Original matrix should remain unchanged
        assert_eq!(get(&matrix, (0, 0)), "a");
        assert_eq!(retrieved, "a_modified");
    }

    #[test]
    fn test_row_clone_independence() {
        let matrix = create_test_matrix();
        let mut row = get_row(&matrix, 0);
        row[0] = 999;

        // Original matrix should remain unchanged
        assert_eq!(get(&matrix, (0, 0)), 1);
        assert_eq!(row[0], 999);
    }

    #[test]
    fn test_col_clone_independence() {
        let matrix = create_test_matrix();
        let mut col = get_col(&matrix, 0);
        col[0] = 999;

        // Original matrix should remain unchanged
        assert_eq!(get(&matrix, (0, 0)), 1);
        assert_eq!(col[0], 999);
    }

    #[test]
    fn test_with_different_types() {
        // Test with f64
        let float_matrix = GenericMatrix {
            data: vec![vec![1.5, 2.5], vec![3.5, 4.5]],
        };
        assert_eq!(get(&float_matrix, (0, 0)), 1.5);
        assert_eq!(get_row(&float_matrix, 0), vec![1.5, 2.5]);
        assert_eq!(get_col(&float_matrix, 0), vec![1.5, 3.5]);

        // Test with bool
        let bool_matrix = GenericMatrix {
            data: vec![vec![true, false], vec![false, true]],
        };
        assert!(get(&bool_matrix, (0, 0)));
        assert_eq!(get_row(&bool_matrix, 0), vec![true, false]);
        assert_eq!(get_col(&bool_matrix, 0), vec![true, false]);
    }
}
