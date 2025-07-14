use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Debug)]
pub struct GenericVector<T> {
    pub data: Vec<T>,
}

impl<T> GenericVector<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    pub fn concat(&self, other: &Self) -> Self {
        let mut data = self.data.clone();
        data.extend(other.data.iter().cloned());
        Self { data }
    }

    fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
        Self { data }
    }

    fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.clone() - b.clone())
            .collect();
        Self { data }
    }

    pub fn dot_product(&self, other: &Self) -> T
    where
        T: std::iter::Sum,
    {
        assert_eq!(self.data.len(), other.data.len());
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a.clone() * b.clone())
            .sum()
    }

    pub fn exterior_product(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        let n = self.data.len();
        let mut result = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let component = self.data[i].clone() * other.data[j].clone()
                    - self.data[j].clone() * other.data[i].clone();
                result.push(component);
            }
        }
        Self { data: result }
    }
}

#[macro_export]
macro_rules! create_vector_interface {
    ($name:ident, $type:ty) => {
        #[pyclass]
        pub struct $name {
            inner: GenericVector<$type>,
        }

        #[pymethods]
        impl $name {
            #[new]
            pub fn new(data: Vec<$type>) -> Self {
                Self {
                    inner: GenericVector::new(data),
                }
            }

            pub fn __getitem__(&self, index: usize) -> PyResult<$type> {
                self.inner
                    .get(index)
                    .cloned()
                    .ok_or_else(|| PyErr::new::<PyIndexError, _>("Index out of bounds"))
            }

            pub fn __len__(&self) -> usize {
                self.inner.data.len()
            }

            pub fn __add__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.add(&other.inner),
                }
            }

            pub fn __sub__(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.sub(&other.inner),
                }
            }

            pub fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($name), self.inner.data)
            }

            pub fn __str__(&self) -> String {
                format!("{:?}", self.inner.data)
            }

            pub fn concat(&self, other: &Self) -> Self {
                Self {
                    inner: self.inner.concat(&other.inner),
                }
            }

            pub fn dot(&self, other: &Self) -> PyResult<$type> {
                if self.inner.data.len() != other.inner.data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Vectors must have the same length for dot product",
                    ));
                }
                Ok(self.inner.dot_product(&other.inner))
            }

            pub fn exterior_product(&self, other: &Self) -> PyResult<Self> {
                if self.inner.data.len() != other.inner.data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Vectors must have the same length for exterior product",
                    ));
                }
                Ok(Self {
                    inner: self.inner.exterior_product(&other.inner),
                })
            }

            pub fn to_list(&self) -> Vec<$type> {
                self.inner.data.clone()
            }

            pub fn dim(&self) -> usize {
                self.inner.data.len()
            }

            pub fn copy(&self) -> Self {
                Self {
                    inner: self.inner.clone(),
                }
            }
        }
    };
}

create_vector_interface!(IntVector, i64);
create_vector_interface!(FloatVector, f64);
