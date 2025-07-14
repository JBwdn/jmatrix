use crate::generic_matrix::GenericMatrix;
use num_traits::{Float, FromPrimitive, One, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Sub};

use crate::ops::{broadcast, ewise, getters, initialisers, matmul, properties, stats};

impl<T> GenericMatrix<T>
where
    T: Clone
        + Default
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + One
        + Zero
        + AddAssign
        + FromPrimitive
        + Float,
{
    // Initialisers:
    pub fn new(data: Vec<Vec<T>>) -> Self {
        initialisers::matrix_from_data(data)
    }

    pub fn filled(shape: (usize, usize), value: T) -> Self {
        initialisers::filled_matrix(shape, value)
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        initialisers::zeros(shape)
    }

    pub fn ones(shape: (usize, usize)) -> Self {
        initialisers::ones(shape)
    }

    pub fn eye(size: usize) -> Self {
        initialisers::eye(size)
    }

    // Properties:
    pub fn shape(&self) -> (usize, usize) {
        properties::shape(self)
    }

    pub fn size(&self) -> usize {
        properties::size(self)
    }

    pub fn len(&self) -> usize {
        properties::len(self)
    }

    pub fn is_empty(&self) -> bool {
        properties::is_empty(self)
    }

    pub fn transpose(&self) -> Self {
        properties::transpose(self)
    }

    pub fn invert(&self) -> Result<Self, String> {
        properties::invert(self)
    }

    // Getters:
    pub fn get(&self, indices: (usize, usize)) -> T {
        getters::get(self, indices)
    }

    pub fn get_row(&self, index: usize) -> Vec<T> {
        getters::get_row(self, index)
    }

    pub fn get_col(&self, index: usize) -> Vec<T> {
        getters::get_col(self, index)
    }

    // Element-wise:
    pub fn ewise_add(&self, other: &Self) -> Self {
        ewise::add(self, other)
    }

    pub fn ewise_sub(&self, other: &Self) -> Self {
        ewise::sub(self, other)
    }

    pub fn ewise_mul(&self, other: &Self) -> Self {
        ewise::mul(self, other)
    }

    pub fn ewise_div(&self, other: &Self) -> Self {
        ewise::div(self, other)
    }

    // Matrix multiplication:
    pub fn matmul(&self, other: &Self) -> Self {
        matmul::matmul(self, other)
    }

    // Broadcasting:
    pub fn broadcast(&self, other_shape: (usize, usize)) -> Self {
        broadcast::broadcast(self, other_shape)
    }

    // Stats:
    pub fn mean(&self, axis: Option<usize>) -> Self {
        stats::mean(self, axis)
    }

    pub fn std(&self, axis: Option<usize>, ddof: Option<usize>) -> Self {
        stats::std(self, axis, ddof)
    }

    pub fn var(&self, axis: Option<usize>, ddof: Option<usize>) -> Self {
        stats::var(self, axis, ddof)
    }
}
