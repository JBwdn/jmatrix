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
