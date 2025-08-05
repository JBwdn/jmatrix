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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_vector_new() {
        let vector = GenericVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_generic_vector_get() {
        let vector = GenericVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.get(0), Some(&1.0));
        assert_eq!(vector.get(1), Some(&2.0));
        assert_eq!(vector.get(2), Some(&3.0));
        assert_eq!(vector.get(3), None);
    }

    #[test]
    fn test_generic_vector_concat() {
        let vector1 = GenericVector::new(vec![1.0, 2.0]);
        let vector2 = GenericVector::new(vec![3.0, 4.0]);
        let result = vector1.concat(&vector2);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_generic_vector_add() {
        let vector1 = GenericVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = GenericVector::new(vec![4.0, 5.0, 6.0]);
        let result = vector1.add(&vector2);
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_generic_vector_sub() {
        let vector1 = GenericVector::new(vec![5.0, 7.0, 9.0]);
        let vector2 = GenericVector::new(vec![1.0, 2.0, 3.0]);
        let result = vector1.sub(&vector2);
        assert_eq!(result.data, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_generic_vector_dot_product() {
        let vector1 = GenericVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = GenericVector::new(vec![4.0, 5.0, 6.0]);
        let result = vector1.dot_product(&vector2);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_generic_vector_dot_product_orthogonal() {
        let vector1 = GenericVector::new(vec![1.0, 0.0]);
        let vector2 = GenericVector::new(vec![0.0, 1.0]);
        let result = vector1.dot_product(&vector2);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_generic_vector_dot_product_single_element() {
        let vector1 = GenericVector::new(vec![3.0]);
        let vector2 = GenericVector::new(vec![4.0]);
        let result = vector1.dot_product(&vector2);
        assert_eq!(result, 12.0);
    }

    #[test]
    fn test_generic_vector_exterior_product_2d() {
        let vector1 = GenericVector::new(vec![1.0, 2.0]);
        let vector2 = GenericVector::new(vec![3.0, 4.0]);
        let result = vector1.exterior_product(&vector2);
        // For 2D: [a1, a2] × [b1, b2] = [a1*b2 - a2*b1] = [1*4 - 2*3] = [-2]
        assert_eq!(result.data, vec![-2.0]);
    }

    #[test]
    fn test_generic_vector_exterior_product_3d() {
        let vector1 = GenericVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = GenericVector::new(vec![4.0, 5.0, 6.0]);
        let result = vector1.exterior_product(&vector2);
        // For 3D: components are (a1*b2 - a2*b1), (a1*b3 - a3*b1), (a2*b3 - a3*b2)
        // (1*5 - 2*4) = -3, (1*6 - 3*4) = -6, (2*6 - 3*5) = -3
        assert_eq!(result.data, vec![-3.0, -6.0, -3.0]);
    }

    #[test]
    #[should_panic]
    fn test_generic_vector_add_different_lengths() {
        let vector1 = GenericVector::new(vec![1.0, 2.0]);
        let vector2 = GenericVector::new(vec![3.0, 4.0, 5.0]);
        vector1.add(&vector2);
    }

    #[test]
    #[should_panic]
    fn test_generic_vector_sub_different_lengths() {
        let vector1 = GenericVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = GenericVector::new(vec![4.0, 5.0]);
        vector1.sub(&vector2);
    }

    #[test]
    #[should_panic]
    fn test_generic_vector_dot_product_different_lengths() {
        let vector1 = GenericVector::new(vec![1.0, 2.0]);
        let vector2 = GenericVector::new(vec![3.0, 4.0, 5.0]);
        vector1.dot_product(&vector2);
    }

    #[test]
    #[should_panic]
    fn test_generic_vector_exterior_product_different_lengths() {
        let vector1 = GenericVector::new(vec![1.0, 2.0]);
        let vector2 = GenericVector::new(vec![3.0, 4.0, 5.0]);
        vector1.exterior_product(&vector2);
    }

    #[test]
    fn test_generic_vector_with_integers() {
        let vector1 = GenericVector::new(vec![1, 2, 3]);
        let vector2 = GenericVector::new(vec![4, 5, 6]);
        
        let add_result = vector1.add(&vector2);
        let sub_result = vector2.sub(&vector1);
        let dot_result = vector1.dot_product(&vector2);
        
        assert_eq!(add_result.data, vec![5, 7, 9]);
        assert_eq!(sub_result.data, vec![3, 3, 3]);
        assert_eq!(dot_result, 32); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_generic_vector_concat_empty() {
        let vector1 = GenericVector::new(vec![1.0, 2.0]);
        let vector2 = GenericVector::new(vec![]);
        let result = vector1.concat(&vector2);
        assert_eq!(result.data, vec![1.0, 2.0]);
        
        let result2 = vector2.concat(&vector1);
        assert_eq!(result2.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_generic_vector_single_element() {
        let vector = GenericVector::new(vec![42.0]);
        assert_eq!(vector.get(0), Some(&42.0));
        assert_eq!(vector.get(1), None);
        
        let vector2 = GenericVector::new(vec![3.0]);
        let concat_result = vector.concat(&vector2);
        assert_eq!(concat_result.data, vec![42.0, 3.0]);
    }

    // Test the Python interface types
    #[test]
    fn test_float_vector_creation() {
        let vector = FloatVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.inner.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_int_vector_creation() {
        let vector = IntVector::new(vec![1, 2, 3]);
        assert_eq!(vector.inner.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_float_vector_operations() {
        let vector1 = FloatVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = FloatVector::new(vec![4.0, 5.0, 6.0]);
        
        let add_result = vector1.__add__(&vector2);
        let sub_result = vector2.__sub__(&vector1);
        
        assert_eq!(add_result.inner.data, vec![5.0, 7.0, 9.0]);
        assert_eq!(sub_result.inner.data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_vector_dot_product_interface() {
        let vector1 = FloatVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = FloatVector::new(vec![4.0, 5.0, 6.0]);
        
        let result = vector1.dot(&vector2).unwrap();
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_vector_exterior_product_interface() {
        let vector1 = FloatVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = FloatVector::new(vec![4.0, 5.0, 6.0]);
        
        let result = vector1.exterior_product(&vector2).unwrap();
        assert_eq!(result.inner.data, vec![-3.0, -6.0, -3.0]);
    }

    #[test]
    fn test_vector_length_and_dim() {
        let vector = FloatVector::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vector.__len__(), 4);
        assert_eq!(vector.dim(), 4);
    }

    #[test]
    fn test_vector_to_list() {
        let vector = IntVector::new(vec![1, 2, 3]);
        let list = vector.to_list();
        assert_eq!(list, vec![1, 2, 3]);
    }

    #[test]
    fn test_vector_copy() {
        let vector1 = FloatVector::new(vec![1.0, 2.0, 3.0]);
        let vector2 = vector1.copy();
        assert_eq!(vector1.inner.data, vector2.inner.data);
        
        // They should be independent copies
        assert_ne!(
            vector1.inner.data.as_ptr(),
            vector2.inner.data.as_ptr()
        );
    }

    #[test]
    fn test_vector_getitem() {
        let vector = FloatVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(vector.__getitem__(0).unwrap(), 1.0);
        assert_eq!(vector.__getitem__(1).unwrap(), 2.0);
        assert_eq!(vector.__getitem__(2).unwrap(), 3.0);
        assert!(vector.__getitem__(3).is_err());
    }

    #[test]
    fn test_vector_concat_interface() {
        let vector1 = IntVector::new(vec![1, 2]);
        let vector2 = IntVector::new(vec![3, 4]);
        let result = vector1.concat(&vector2);
        assert_eq!(result.inner.data, vec![1, 2, 3, 4]);
    }
}
