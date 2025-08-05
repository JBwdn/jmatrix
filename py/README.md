# JMatrix Python API

A comprehensive Python API for the high-performance JMatrix library, providing NumPy-like functionality for floating-point matrix operations with Rust-powered performance.

## Features

- **High Performance**: Rust-based implementation for fast matrix operations
- **NumPy-like Interface**: Familiar API for Python developers
- **Comprehensive Operations**: Element-wise operations, matrix multiplication, statistical functions
- **Broadcasting Support**: Automatic shape broadcasting for operations
- **Type Safety**: Built-in type checking and validation
- **Full Documentation**: Comprehensive docstrings and examples

## Installation

Build and install from source:

```bash
# Install maturin (build tool)
pip install maturin

# Build and install the package
maturin develop  # For development
# or
maturin build && pip install target/wheels/*.whl  # For production
```

## Quick Start

```python
import jmatrix as jm

# Create matrices
m1 = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
m2 = jm.zeros((2, 2))
identity = jm.eye(2)

# Basic operations
result = m1 + m2
product = m1 @ identity  # Matrix multiplication
transposed = m1.T

# Properties and access
print(f"Shape: {m1.shape()}")       # (2, 2)
print(f"Element: {m1[0, 1]}")       # 2.0
print(f"Row: {m1.get_row(0)}")      # [1.0, 2.0]

# Aggregations
total = m1.sum()                     # 10.0
col_sums = m1.sum(axis=0)           # Column sums
row_maxes = m1.max(axis=1)          # Row maxes
```

## API Reference

### Matrix Creation

#### Constructors

```python
# From nested lists
m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])

# Convenience functions
zeros = jm.zeros((2, 3))        # 2x3 matrix of zeros
ones = jm.ones((2, 2))          # 2x2 matrix of ones  
eye = jm.eye(3)                 # 3x3 identity matrix
```

#### Alternative Creation Methods

```python
# From lists (aliases)
m = jm.from_lists([[1.0, 2.0], [3.0, 4.0]])
m = jm.matrix([[1.0, 2.0], [3.0, 4.0]])
```

### Properties and Metadata

```python
m = jm.FloatMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Shape information
m.shape()           # (2, 3) - (rows, cols)
m.ndim              # 2 - number of dimensions
m.rows              # 2 - number of rows
m.cols              # 3 - number of columns
m.size()            # 6 - total elements
m.len()             # 2 - number of rows
m.is_empty()        # False

# Data access
m.data()            # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
m.tolist()          # Same as data()
m.flatten()         # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

### Element Access and Indexing

```python
m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])

# Individual elements
m[0, 1]             # 2.0
m.get((0, 1))       # 2.0

# Rows and columns
m.get_row(0)        # [1.0, 2.0]
m.get_col(1)        # [2.0, 4.0]
```

### Mathematical Operations

#### Element-wise Operations

```python
m1 = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
m2 = jm.FloatMatrix([[5.0, 6.0], [7.0, 8.0]])

# Operators (with broadcasting)
m1 + m2             # Element-wise addition
m1 - m2             # Element-wise subtraction
m1 * m2             # Element-wise multiplication
m1 / m2             # Element-wise division

# Explicit methods
m1.ewise_add(m2)    # Same as m1 + m2
m1.ewise_sub(m2)    # Same as m1 - m2
m1.ewise_mul(m2)    # Same as m1 * m2
m1.ewise_div(m2)    # Same as m1 / m2
```

#### Matrix Operations

```python
# Matrix multiplication
m1 @ m2             # Using @ operator
m1.matmul(m2)       # Explicit method

# Transpose
m1.T                # Property
m1.transpose()      # Method

# Matrix inversion (for square matrices)
m1.invert()         # Returns inverse matrix

# Broadcasting
small = jm.FloatMatrix([[1.0, 2.0]])
broadcasted = small.broadcast((3, 2))  # Repeat to 3x2
```

### Aggregation Functions

```python
m = jm.FloatMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Sum
m.sum()             # 21.0 - total sum
m.sum(axis=0)       # [[5.0, 7.0, 9.0]] - column sums
m.sum(axis=1)       # [[6.0], [15.0]] - row sums

# Maximum
m.max()             # 6.0 - global maximum
m.max(axis=0)       # [[4.0, 5.0, 6.0]] - column maxes
m.max(axis=1)       # [[3.0], [6.0]] - row maxes

# Minimum
m.min()             # 1.0 - global minimum
m.min(axis=0)       # [[1.0, 2.0, 3.0]] - column mins
m.min(axis=1)       # [[1.0], [4.0]] - row mins
```

### Statistical Functions

```python
m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])

# Built-in statistical functions
m.mean()            # Mean (returns FloatMatrix)
m.mean(0)           # Mean along axis 0
m.mean(1)           # Mean along axis 1

m.std()             # Standard deviation
m.std(0, 1)         # Std along axis 0, ddof=1

m.var()             # Variance
m.var(1, 0)         # Variance along axis 1, ddof=0
```

### Utility Functions

```python
m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])

# Copy
m_copy = m.copy()   # Create a deep copy

# String representations
str(m)              # "[[1.0, 2.0], [3.0, 4.0]]"
repr(m)             # "FloatMatrix([[1.0, 2.0], [3.0, 4.0]])"
```

## Advanced Examples

### Linear Algebra Operations

```python
import jmatrix as jm

# Solve a simple linear system: Ax = b
A = jm.FloatMatrix([[2.0, 1.0], [1.0, 3.0]])
b = jm.FloatMatrix([[5.0], [6.0]])

# x = A^-1 * b (note: inversion may have numerical issues)
try:
    A_inv = A.invert()
    x = A_inv @ b
    print(f"Solution: {x}")
except:
    print("Matrix inversion failed")
```

### Statistical Analysis

```python
import jmatrix as jm

# Create sample data
data = jm.FloatMatrix([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0], 
    [7.0, 8.0, 9.0]
])

# Statistical summary
print(f"Data:\n{data}")
print(f"Mean: {data.mean()}")
print(f"Column means: {data.mean(axis=0)}")
print(f"Row means: {data.mean(axis=1)}")
print(f"Standard deviation: {data.std()}")
print(f"Min: {data.min()}, Max: {data.max()}")
```

### Broadcasting and Operations

```python
import jmatrix as jm

# Broadcasting example
matrix = jm.FloatMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
row_vector = jm.FloatMatrix([[10.0, 20.0, 30.0]])

# Broadcasting happens automatically in operations
result = matrix + row_vector
print(f"Broadcasted addition:\n{result}")

# Manual broadcasting
broadcasted = row_vector.broadcast(matrix.shape())
explicit_result = matrix + broadcasted
print(f"Explicit broadcasting:\n{explicit_result}")
```

## Performance Considerations

- **Memory Layout**: Matrices use row-major storage for compatibility with Python/NumPy
- **Operations**: All mathematical operations are implemented in Rust for optimal performance
- **Broadcasting**: Automatic broadcasting minimizes memory allocation
- **Type Safety**: All operations are type-checked at the Rust level

## Compatibility

- **Python**: Requires Python 3.8+
- **Dependencies**: No runtime Python dependencies (pure Rust implementation)
- **Platforms**: Supports Linux, macOS, and Windows

## Contributing

This is a Python wrapper around a Rust-based matrix library. The Python API is designed to be:

1. **Familiar**: NumPy-like interface for Python developers
2. **Complete**: Full access to underlying Rust functionality  
3. **Documented**: Comprehensive docstrings and examples
4. **Tested**: Extensive test coverage for all functionality

## License

See the main project license for details.