"""
Tests for the jmatrix Python API.

These tests validate the enhanced Python interface for the FloatMatrix class,
ensuring that all functionality works correctly and maintains backward compatibility.
"""

try:
    import pytest
except ImportError:
    # pytest not available, will use manual test runner
    pytest = None

import jmatrix as jm


class TestFloatMatrixCreation:
    """Test matrix creation methods."""
    
    def test_constructor(self):
        """Test basic constructor."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.shape() == (2, 2)
        assert m.data() == [[1.0, 2.0], [3.0, 4.0]]
    
    def test_zeros(self):
        """Test zeros constructor."""
        zeros = jm.zeros((2, 3))
        assert zeros.shape() == (2, 3)
        assert zeros.data() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
    def test_ones(self):
        """Test ones constructor."""
        ones = jm.ones((2, 2))
        assert ones.shape() == (2, 2)
        assert ones.data() == [[1.0, 1.0], [1.0, 1.0]]
    
    def test_eye(self):
        """Test identity matrix constructor."""
        eye = jm.eye(3)
        assert eye.shape() == (3, 3)
        expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert eye.data() == expected
    
    def test_from_lists(self):
        """Test from_lists constructor."""
        m = jm.from_lists([[1.0, 2.0], [3.0, 4.0]])
        assert m.shape() == (2, 2)
        assert m.data() == [[1.0, 2.0], [3.0, 4.0]]
    
    def test_matrix(self):
        """Test matrix constructor."""
        m = jm.matrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.shape() == (2, 2)
        assert m.data() == [[1.0, 2.0], [3.0, 4.0]]


class TestFloatMatrixProperties:
    """Test matrix properties and metadata."""
    
    def test_basic_properties(self):
        """Test shape, size, len, etc."""
        m = jm.FloatMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert m.shape() == (2, 3)
        assert m.size() == 6
        assert m.len() == 2
        assert not m.is_empty()
        assert m.ndim == 2
        assert m.rows == 2
        assert m.cols == 3
    
    def test_empty_matrix(self):
        """Test empty matrix properties."""
        empty = jm.FloatMatrix([])
        assert empty.is_empty()
    
    def test_data_access(self):
        """Test data access methods."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.data() == [[1.0, 2.0], [3.0, 4.0]]
        assert m.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert m.flatten() == [1.0, 2.0, 3.0, 4.0]


class TestFloatMatrixIndexing:
    """Test matrix indexing and element access."""
    
    def test_get_element(self):
        """Test element access."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.get((0, 0)) == 1.0
        assert m.get((0, 1)) == 2.0
        assert m.get((1, 0)) == 3.0
        assert m.get((1, 1)) == 4.0
    
    def test_getitem(self):
        """Test __getitem__ access."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        assert m[0, 0] == 1.0
        assert m[0, 1] == 2.0
        assert m[1, 0] == 3.0
        assert m[1, 1] == 4.0
    
    def test_get_row_col(self):
        """Test row and column access."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        assert m.get_row(0) == [1.0, 2.0]
        assert m.get_row(1) == [3.0, 4.0]
        assert m.get_col(0) == [1.0, 3.0]
        assert m.get_col(1) == [2.0, 4.0]


class TestFloatMatrixOperations:
    """Test matrix mathematical operations."""
    
    def test_element_wise_operations(self):
        """Test element-wise operations."""
        m1 = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = jm.FloatMatrix([[5.0, 6.0], [7.0, 8.0]])
        
        # Addition
        result = m1 + m2
        assert result.data() == [[6.0, 8.0], [10.0, 12.0]]
        
        # Subtraction  
        result = m1 - m2
        assert result.data() == [[-4.0, -4.0], [-4.0, -4.0]]
        
        # Multiplication
        result = m1 * m2
        assert result.data() == [[5.0, 12.0], [21.0, 32.0]]
        
        # Division
        result = m2 / m1
        assert result.data() == [[5.0, 3.0], [7.0/3.0, 2.0]]
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        m1 = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        m2 = jm.FloatMatrix([[5.0, 6.0], [7.0, 8.0]])
        
        result = m1 @ m2
        expected = [[19.0, 22.0], [43.0, 50.0]]
        assert result.data() == expected
        
        # Also test explicit matmul method
        result2 = m1.matmul(m2)
        assert result2.data() == expected
    
    def test_transpose(self):
        """Test matrix transpose."""
        m = jm.FloatMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Test explicit method
        t1 = m.transpose()
        expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        assert t1.data() == expected
        
        # Test property
        t2 = m.T
        assert t2.data() == expected


class TestFloatMatrixAggregation:
    """Test matrix aggregation functions."""
    
    def test_sum(self):
        """Test sum operations."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        
        # Total sum
        assert m.sum() == 10.0
        
        # Sum along axis 0 (column sums)
        col_sums = m.sum(axis=0)
        assert col_sums.data() == [[4.0, 6.0]]
        
        # Sum along axis 1 (row sums)
        row_sums = m.sum(axis=1)
        assert row_sums.data() == [[3.0], [7.0]]
    
    def test_max_min(self):
        """Test max and min operations."""
        m = jm.FloatMatrix([[1.0, 4.0], [2.0, 3.0]])
        
        # Global max/min
        assert m.max() == 4.0
        assert m.min() == 1.0
        
        # Max along axis 0 (column maxes)
        col_max = m.max(axis=0)
        assert col_max.data() == [[2.0, 4.0]]
        
        # Min along axis 1 (row mins)
        row_min = m.min(axis=1)
        assert row_min.data() == [[1.0], [2.0]]


class TestFloatMatrixUtilities:
    """Test utility functions."""
    
    def test_copy(self):
        """Test matrix copying."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        m_copy = m.copy()
        
        # Should have same data
        assert m_copy.data() == m.data()
        
        # Should be different objects
        assert m is not m_copy
    
    def test_string_representations(self):
        """Test string representations."""
        m = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
        
        assert str(m) == "[[1.0, 2.0], [3.0, 4.0]]"
        assert repr(m) == "FloatMatrix([[1.0, 2.0], [3.0, 4.0]])"


class TestFloatMatrixStats:
    """Test statistical functions."""
    
    def test_mean(self):
        """Test mean calculations."""
        m = jm.FloatMatrix([[2.0, 4.0], [6.0, 8.0]])
        
        # Mean without axis (should return matrix)
        mean_result = m.mean()
        assert isinstance(mean_result, jm.FloatMatrix)
        
        # Mean along axes
        mean_axis0 = m.mean(0)
        assert isinstance(mean_axis0, jm.FloatMatrix)
        
        mean_axis1 = m.mean(1)
        assert isinstance(mean_axis1, jm.FloatMatrix)


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    import sys
    
    # Create test instances
    test_classes = [
        TestFloatMatrixCreation(),
        TestFloatMatrixProperties(),
        TestFloatMatrixIndexing(),
        TestFloatMatrixOperations(),
        TestFloatMatrixAggregation(),
        TestFloatMatrixUtilities(),
        TestFloatMatrixStats(),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_instance in test_classes:
        methods = [method for method in dir(test_instance) if method.startswith('test_')]
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {test_instance.__class__.__name__}.{method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {test_instance.__class__.__name__}.{method_name}: {e}")
    
    print(f"\nTests passed: {passed_tests}/{total_tests}")
    if passed_tests == total_tests:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)