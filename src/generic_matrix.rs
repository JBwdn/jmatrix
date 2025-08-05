#[derive(Clone, Debug)]
pub struct GenericMatrix<T> {
    pub data: Vec<Vec<T>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_matrix_creation() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let matrix = GenericMatrix { data: data.clone() };
        assert_eq!(matrix.data, data);
    }

    #[test]
    fn test_generic_matrix_clone() {
        let matrix1 = GenericMatrix {
            data: vec![vec![1, 2], vec![3, 4]],
        };
        let matrix2 = matrix1.clone();
        
        assert_eq!(matrix1.data, matrix2.data);
        // Verify they are independent copies
        assert_ne!(matrix1.data.as_ptr(), matrix2.data.as_ptr());
    }

    #[test]
    fn test_generic_matrix_debug() {
        let matrix = GenericMatrix {
            data: vec![vec![1, 2], vec![3, 4]],
        };
        let debug_str = format!("{:?}", matrix);
        assert!(debug_str.contains("GenericMatrix"));
        assert!(debug_str.contains("[[1, 2], [3, 4]]"));
    }

    #[test]
    fn test_generic_matrix_empty() {
        let matrix: GenericMatrix<f64> = GenericMatrix { data: vec![] };
        assert!(matrix.data.is_empty());
    }

    #[test]
    fn test_generic_matrix_single_element() {
        let matrix = GenericMatrix { data: vec![vec![42]] };
        assert_eq!(matrix.data[0][0], 42);
        assert_eq!(matrix.data.len(), 1);
        assert_eq!(matrix.data[0].len(), 1);
    }

    #[test]
    fn test_generic_matrix_different_types() {
        // Test with floats
        let float_matrix = GenericMatrix {
            data: vec![vec![1.5, 2.7], vec![3.14, 4.0]],
        };
        assert_eq!(float_matrix.data[0][0], 1.5);

        // Test with strings
        let string_matrix = GenericMatrix {
            data: vec![vec!["hello".to_string(), "world".to_string()]],
        };
        assert_eq!(string_matrix.data[0][0], "hello");

        // Test with booleans
        let bool_matrix = GenericMatrix {
            data: vec![vec![true, false], vec![false, true]],
        };
        assert_eq!(bool_matrix.data[0][0], true);
    }
}
