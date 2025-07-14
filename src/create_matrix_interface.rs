macro_rules! create_matrix_interface {
    ($name:ident, $type:ty) => {
        #[pyclass]
        pub struct $name {
            inner: GenericMatrix<$type>,
        }

        #[pymethods]
        impl $name {
            // Initialisers:
            #[staticmethod]
            pub fn from_lists(data: Vec<Vec<$type>>) -> Self {
                Self {
                    inner: GenericMatrix::new(data),
                }
            }

            #[staticmethod]
            pub fn zeros(shape: (usize, usize)) -> Self {
                Self {
                    inner: GenericMatrix::zeros(shape),
                }
            }

            #[staticmethod]
            pub fn ones(shape: (usize, usize)) -> Self {
                Self {
                    inner: GenericMatrix::ones(shape),
                }
            }

            #[staticmethod]
            pub fn eye(size: usize) -> Self {
                Self {
                    inner: GenericMatrix::eye(size),
                }
            }

            // Properties:
            pub fn shape(&self) -> (usize, usize) {
                self.inner.shape()
            }

            pub fn size(&self) -> usize {
                self.inner.size()
            }

            pub fn len(&self) -> usize {
                self.inner.len()
            }

            pub fn is_empty(&self) -> bool {
                self.inner.is_empty()
            }

            pub fn data(&self) -> Vec<Vec<$type>> {
                self.inner.data.clone()
            }

            pub fn transpose(&self) -> Self {
                Self {
                    inner: self.inner.transpose(),
                }
            }

            pub fn invert(&self) -> PyResult<Self> {
                match self.inner.invert() {
                    Ok(inverted) => Ok(Self { inner: inverted }),
                    Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e)),
                }
            }

            // Getters:
            pub fn get(&self, indices: (usize, usize)) -> $type {
                self.inner.get(indices)
            }

            pub fn get_row(&self, index: usize) -> Vec<$type> {
                self.inner.get_row(index)
            }

            pub fn get_col(&self, index: usize) -> Vec<$type> {
                self.inner.get_col(index)
            }

            // Element-wise:
            pub fn ewise_add(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.ewise_add(&other.inner),
                }
            }

            pub fn ewise_sub(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.ewise_sub(&other.inner),
                }
            }

            pub fn ewise_div(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.ewise_div(&other.inner),
                }
            }

            pub fn ewise_mul(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.ewise_mul(&other.inner),
                }
            }

            // Matrix multiplication:
            pub fn matmul(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.matmul(&other.inner),
                }
            }

            // Broadcasting:
            pub fn broadcast(&self, other_shape: (usize, usize)) -> Self {
                Self {
                    inner: self.inner.broadcast(other_shape),
                }
            }

            // Stats:
            pub fn mean(&self, axis: Option<usize>) -> Self {
                Self {
                    inner: self.inner.mean(axis),
                }
            }

            pub fn std(&self, axis: Option<usize>, ddof: Option<usize>) -> Self {
                Self {
                    inner: self.inner.std(axis, ddof),
                }
            }

            pub fn var(&self, axis: Option<usize>, ddof: Option<usize>) -> Self {
                Self {
                    inner: self.inner.var(axis, ddof),
                }
            }

            // Python special methods:
            #[new]
            pub fn new(data: Vec<Vec<$type>>) -> Self {
                Self {
                    inner: GenericMatrix::new(data),
                }
            }

            pub fn __getitem__(&self, indices: (usize, usize)) -> $type {
                self.inner.get(indices)
            }

            pub fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($name), self.inner.data)
            }

            pub fn __str__(&self) -> String {
                format!("{:?}", self.inner.data)
            }

            pub fn __matmul__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.matmul(&other.inner),
                }
            }

            pub fn __add__(&self, other: &Self) -> Self {
                let other_b = other.inner.broadcast(self.shape());
                Self {
                    inner: self.inner.ewise_add(&other_b),
                }
            }

            pub fn __sub__(&self, other: &Self) -> Self {
                let other_b = other.inner.broadcast(self.shape());
                Self {
                    inner: self.inner.ewise_sub(&other_b),
                }
            }

            pub fn __mul__(&self, other: &Self) -> Self {
                let other_b = other.inner.broadcast(self.shape());
                Self {
                    inner: self.inner.ewise_mul(&other_b),
                }
            }

            pub fn __truediv__(&self, other: &Self) -> Self {
                let other_b = other.inner.broadcast(self.shape());
                Self {
                    inner: self.inner.ewise_div(&other_b),
                }
            }

            #[getter]
            #[allow(non_snake_case)]
            pub fn T(&self) -> Self {
                Self {
                    inner: self.inner.transpose(),
                }
            }
        }
    };
}
