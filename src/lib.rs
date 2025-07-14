use pyo3::prelude::*;

pub mod generic_matrix;
use crate::generic_matrix::GenericMatrix;

#[macro_use]
pub mod create_matrix_interface;

pub mod generic_matrix_methods;

pub mod ops;

create_matrix_interface!(FloatMatrix, f64);

#[pymodule]
#[pyo3(name = "_lib")]
fn my_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FloatMatrix>()?;
    Ok(())
}
