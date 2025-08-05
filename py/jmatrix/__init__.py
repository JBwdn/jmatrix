"""
JMatrix: A high-performance matrix library with Python bindings.

This module provides a comprehensive Python API for working with floating-point matrices,
built on top of a fast Rust implementation.

Main Classes:
    FloatMatrix: A 2D matrix of float64 values with comprehensive mathematical operations.

Key Features:
    - Fast matrix operations (element-wise, matrix multiplication, etc.)
    - NumPy-like interface and broadcasting
    - Statistical functions (mean, std, var)
    - Convenient constructors and property access
    - Full Python operator overloading
"""

from typing import List, Tuple, Optional, Union
from jmatrix._lib import FloatMatrix as _FloatMatrix

__version__ = "0.1.0"
__all__ = ["FloatMatrix", "zeros", "ones", "eye", "from_lists", "matrix"]


class FloatMatrix:
    """
    A 2D matrix of float64 values with comprehensive mathematical operations.
    
    This class provides a NumPy-like interface for matrix operations while being
    backed by high-performance Rust code.
    
    Attributes:
        T: Transpose of the matrix (property)
        
    Examples:
        >>> # Create matrices
        >>> m = FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        >>> zeros = FloatMatrix.zeros((2, 3))
        >>> eye = FloatMatrix.eye(3)
        
        >>> # Basic operations
        >>> m.shape()
        (2, 2)
        >>> m[0, 1]
        2.0
        >>> m.T
        FloatMatrix([[1.0, 3.0], [2.0, 4.0]])
        
        >>> # Matrix operations
        >>> m1 @ m2  # matrix multiplication
        >>> m1 + m2  # element-wise addition
        >>> m1 * m2  # element-wise multiplication
        
        >>> # Statistics
        >>> m.mean()
        >>> m.std()
    """
    
    def __init__(self, data: List[List[float]]):
        """
        Create a matrix from nested lists of floats.
        
        Args:
            data: A list of lists representing the matrix rows.
                 All inner lists must have the same length.
                 
        Examples:
            >>> m = FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
            >>> m.shape()
            (2, 2)
        """
        self._matrix = _FloatMatrix(data)
    
    @classmethod 
    def _from_rust_matrix(cls, rust_matrix: _FloatMatrix) -> 'FloatMatrix':
        """Create a FloatMatrix from an existing Rust matrix (internal use)."""
        obj = cls.__new__(cls)
        obj._matrix = rust_matrix
        return obj
    
    # Delegate all original methods to the Rust implementation
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the matrix as (rows, cols)."""
        return self._matrix.shape()
    
    def size(self) -> int:
        """Get the total number of elements in the matrix."""
        return self._matrix.size()
    
    def len(self) -> int:
        """Get the number of rows in the matrix."""
        return self._matrix.len()
    
    def is_empty(self) -> bool:
        """Check if the matrix is empty."""
        return self._matrix.is_empty()
    
    def data(self) -> List[List[float]]:
        """Get the matrix data as nested lists."""
        return self._matrix.data()
    
    def transpose(self) -> 'FloatMatrix':
        """Return the transpose of the matrix."""
        return FloatMatrix._from_rust_matrix(self._matrix.transpose())
    
    def invert(self) -> 'FloatMatrix':
        """Return the inverse of the matrix."""
        return FloatMatrix._from_rust_matrix(self._matrix.invert())
    
    def get(self, indices: Tuple[int, int]) -> float:
        """Get element at specified indices."""
        return self._matrix.get(indices)
    
    def get_row(self, index: int) -> List[float]:
        """Get a row as a list."""
        return self._matrix.get_row(index)
    
    def get_col(self, index: int) -> List[float]:
        """Get a column as a list."""
        return self._matrix.get_col(index)
    
    def ewise_add(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise addition."""
        return FloatMatrix._from_rust_matrix(self._matrix.ewise_add(other._matrix))
    
    def ewise_sub(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise subtraction."""
        return FloatMatrix._from_rust_matrix(self._matrix.ewise_sub(other._matrix))
    
    def ewise_div(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise division."""
        return FloatMatrix._from_rust_matrix(self._matrix.ewise_div(other._matrix))
    
    def ewise_mul(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise multiplication."""
        return FloatMatrix._from_rust_matrix(self._matrix.ewise_mul(other._matrix))
    
    def matmul(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Matrix multiplication."""
        return FloatMatrix._from_rust_matrix(self._matrix.matmul(other._matrix))
    
    def broadcast(self, other_shape: Tuple[int, int]) -> 'FloatMatrix':
        """Broadcast matrix to a new shape."""
        return FloatMatrix._from_rust_matrix(self._matrix.broadcast(other_shape))
    
    def mean(self, axis: Optional[int] = None) -> 'FloatMatrix':
        """Compute the mean along the specified axis."""
        return FloatMatrix._from_rust_matrix(self._matrix.mean(axis))
    
    def std(self, axis: Optional[int] = None, ddof: Optional[int] = None) -> 'FloatMatrix':
        """Compute the standard deviation along the specified axis."""
        return FloatMatrix._from_rust_matrix(self._matrix.std(axis, ddof))
    
    def var(self, axis: Optional[int] = None, ddof: Optional[int] = None) -> 'FloatMatrix':
        """Compute the variance along the specified axis."""
        return FloatMatrix._from_rust_matrix(self._matrix.var(axis, ddof))
    
    # Python special methods
    def __getitem__(self, indices: Tuple[int, int]) -> float:
        """Get element using matrix[i, j] syntax."""
        return self._matrix.__getitem__(indices)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"FloatMatrix({self.data()})"
    
    def __str__(self) -> str:
        """String representation for display."""
        return str(self.data())
    
    def __matmul__(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Matrix multiplication using @ operator."""
        return FloatMatrix._from_rust_matrix(self._matrix.__matmul__(other._matrix))
    
    def __add__(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise addition using + operator."""
        return FloatMatrix._from_rust_matrix(self._matrix.__add__(other._matrix))
    
    def __sub__(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise subtraction using - operator."""
        return FloatMatrix._from_rust_matrix(self._matrix.__sub__(other._matrix))
    
    def __mul__(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise multiplication using * operator."""
        return FloatMatrix._from_rust_matrix(self._matrix.__mul__(other._matrix))
    
    def __truediv__(self, other: 'FloatMatrix') -> 'FloatMatrix':
        """Element-wise division using / operator."""
        return FloatMatrix._from_rust_matrix(self._matrix.__truediv__(other._matrix))
    
    @property
    def T(self) -> 'FloatMatrix':
        """Transpose of the matrix."""
        return FloatMatrix._from_rust_matrix(self._matrix.T)
    
    @classmethod
    def zeros(cls, shape: Tuple[int, int]) -> 'FloatMatrix':
        """
        Create a matrix filled with zeros.
        
        Args:
            shape: Tuple of (rows, columns)
            
        Returns:
            A new FloatMatrix filled with zeros
            
        Examples:
            >>> zeros = FloatMatrix.zeros((2, 3))
            >>> zeros.shape()
            (2, 3)
        """
        return cls._from_rust_matrix(_FloatMatrix.zeros(shape))
    
    @classmethod
    def ones(cls, shape: Tuple[int, int]) -> 'FloatMatrix':
        """
        Create a matrix filled with ones.
        
        Args:
            shape: Tuple of (rows, columns)
            
        Returns:
            A new FloatMatrix filled with ones
            
        Examples:
            >>> ones = FloatMatrix.ones((2, 2))
            >>> ones.data()
            [[1.0, 1.0], [1.0, 1.0]]
        """
        return cls._from_rust_matrix(_FloatMatrix.ones(shape))
    
    @classmethod
    def eye(cls, size: int) -> 'FloatMatrix':
        """
        Create an identity matrix.
        
        Args:
            size: Size of the square identity matrix
            
        Returns:
            A new identity FloatMatrix
            
        Examples:
            >>> eye = FloatMatrix.eye(3)
            >>> eye.data()
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """
        return cls._from_rust_matrix(_FloatMatrix.eye(size))
    
    @classmethod
    def from_lists(cls, data: List[List[float]]) -> 'FloatMatrix':
        """
        Create a matrix from nested lists (alias for constructor).
        
        Args:
            data: A list of lists representing the matrix rows
            
        Returns:
            A new FloatMatrix
            
        Examples:
            >>> m = FloatMatrix.from_lists([[1.0, 2.0], [3.0, 4.0]])
        """
        return cls._from_rust_matrix(_FloatMatrix.from_lists(data))
    
    def tolist(self) -> List[List[float]]:
        """
        Convert matrix to nested Python lists.
        
        Returns:
            The matrix data as nested lists
            
        Examples:
            >>> m = FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
            >>> m.tolist()
            [[1.0, 2.0], [3.0, 4.0]]
        """
        return self.data()
    
    @property
    def ndim(self) -> int:
        """Number of dimensions (always 2 for matrices)."""
        return 2
    
    @property
    def rows(self) -> int:
        """Number of rows in the matrix."""
        return self.shape()[0]
    
    @property
    def cols(self) -> int:
        """Number of columns in the matrix."""
        return self.shape()[1]
    
    def copy(self) -> 'FloatMatrix':
        """
        Create a copy of the matrix.
        
        Returns:
            A new FloatMatrix with the same data
        """
        return FloatMatrix(self.data())
    
    def flatten(self) -> List[float]:
        """
        Return a flattened (1D) list of all matrix elements in row-major order.
        
        Returns:
            A flat list of all matrix elements
            
        Examples:
            >>> m = FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
            >>> m.flatten()
            [1.0, 2.0, 3.0, 4.0]
        """
        result = []
        for row in self.data():
            result.extend(row)
        return result
    
    def sum(self, axis: Optional[int] = None) -> Union['FloatMatrix', float]:
        """
        Sum of matrix elements.
        
        Args:
            axis: Axis along which to sum. None for total sum,
                 0 for column sums, 1 for row sums
                 
        Returns:
            Sum as float (if axis=None) or FloatMatrix (if axis specified)
            
        Examples:
            >>> m = FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
            >>> m.sum()
            10.0
            >>> m.sum(axis=0)  # column sums
            FloatMatrix([[4.0, 6.0]])
        """
        if axis is None:
            return sum(self.flatten())
        
        if axis == 0:  # sum along rows (column sums)
            shape = self.shape()
            result = [[0.0] * shape[1]]
            for col in range(shape[1]):
                col_data = self.get_col(col)
                result[0][col] = sum(col_data)
            return FloatMatrix(result)
        
        elif axis == 1:  # sum along columns (row sums)
            shape = self.shape()
            result = []
            for row in range(shape[0]):
                row_data = self.get_row(row)
                result.append([sum(row_data)])
            return FloatMatrix(result)
        
        else:
            raise ValueError(f"axis {axis} is out of bounds for array of dimension 2")
    
    def max(self, axis: Optional[int] = None) -> Union['FloatMatrix', float]:
        """
        Maximum of matrix elements.
        
        Args:
            axis: Axis along which to find max. None for global max,
                 0 for column maxes, 1 for row maxes
                 
        Returns:
            Max as float (if axis=None) or FloatMatrix (if axis specified)
        """
        if axis is None:
            return max(self.flatten())
        
        if axis == 0:  # max along rows (column maxes)
            shape = self.shape()
            result = [[0.0] * shape[1]]
            for col in range(shape[1]):
                col_data = self.get_col(col)
                result[0][col] = max(col_data)
            return FloatMatrix(result)
        
        elif axis == 1:  # max along columns (row maxes)
            shape = self.shape()
            result = []
            for row in range(shape[0]):
                row_data = self.get_row(row)
                result.append([max(row_data)])
            return FloatMatrix(result)
        
        else:
            raise ValueError(f"axis {axis} is out of bounds for array of dimension 2")
    
    def min(self, axis: Optional[int] = None) -> Union['FloatMatrix', float]:
        """
        Minimum of matrix elements.
        
        Args:
            axis: Axis along which to find min. None for global min,
                 0 for column mins, 1 for row mins
                 
        Returns:
            Min as float (if axis=None) or FloatMatrix (if axis specified)
        """
        if axis is None:
            return min(self.flatten())
        
        if axis == 0:  # min along rows (column mins)
            shape = self.shape()
            result = [[0.0] * shape[1]]
            for col in range(shape[1]):
                col_data = self.get_col(col)
                result[0][col] = min(col_data)
            return FloatMatrix(result)
        
        elif axis == 1:  # min along columns (row mins)
            shape = self.shape()
            result = []
            for row in range(shape[0]):
                row_data = self.get_row(row)
                result.append([min(row_data)])
            return FloatMatrix(result)
        
        else:
            raise ValueError(f"axis {axis} is out of bounds for array of dimension 2")


# Convenience functions for creating matrices
def zeros(shape: Tuple[int, int]) -> FloatMatrix:
    """
    Create a matrix filled with zeros.
    
    Args:
        shape: Tuple of (rows, columns)
        
    Returns:
        A new FloatMatrix filled with zeros
        
    Examples:
        >>> import jmatrix as jm
        >>> zeros = jm.zeros((2, 3))
    """
    return FloatMatrix.zeros(shape)


def ones(shape: Tuple[int, int]) -> FloatMatrix:
    """
    Create a matrix filled with ones.
    
    Args:
        shape: Tuple of (rows, columns)
        
    Returns:
        A new FloatMatrix filled with ones
        
    Examples:
        >>> import jmatrix as jm
        >>> ones = jm.ones((2, 2))
    """
    return FloatMatrix.ones(shape)


def eye(size: int) -> FloatMatrix:
    """
    Create an identity matrix.
    
    Args:
        size: Size of the square identity matrix
        
    Returns:
        A new identity FloatMatrix
        
    Examples:
        >>> import jmatrix as jm
        >>> eye = jm.eye(3)
    """
    return FloatMatrix.eye(size)


def from_lists(data: List[List[float]]) -> FloatMatrix:
    """
    Create a matrix from nested lists.
    
    Args:
        data: A list of lists representing the matrix rows
        
    Returns:
        A new FloatMatrix
        
    Examples:
        >>> import jmatrix as jm
        >>> m = jm.from_lists([[1.0, 2.0], [3.0, 4.0]])
    """
    return FloatMatrix.from_lists(data)


def matrix(data: List[List[float]]) -> FloatMatrix:
    """
    Create a matrix from nested lists (alias for from_lists).
    
    Args:
        data: A list of lists representing the matrix rows
        
    Returns:
        A new FloatMatrix
        
    Examples:
        >>> import jmatrix as jm
        >>> m = jm.matrix([[1.0, 2.0], [3.0, 4.0]])
    """
    return FloatMatrix(data)
