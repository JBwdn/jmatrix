"""
Example usage of the jmatrix Python API.

This script demonstrates the key features of the enhanced FloatMatrix API,
showing how to create matrices, perform operations, and use the NumPy-like interface.
"""

import jmatrix as jm

def main():
    print("JMatrix Python API Demo")
    print("=" * 40)
    
    # Matrix Creation
    print("\n1. Matrix Creation")
    print("-" * 20)
    
    # Basic constructor
    m1 = jm.FloatMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"From lists: {m1}")
    
    # Convenience constructors
    zeros = jm.zeros((2, 3))
    ones = jm.ones((2, 2))
    eye = jm.eye(3)
    
    print(f"Zeros (2x3): {zeros}")
    print(f"Ones (2x2): {ones}")
    print(f"Identity (3x3): {eye}")
    
    # Matrix Properties
    print("\n2. Matrix Properties")
    print("-" * 20)
    
    print(f"Shape: {m1.shape()}")
    print(f"Dimensions: {m1.ndim}")
    print(f"Rows: {m1.rows}, Cols: {m1.cols}")
    print(f"Size: {m1.size()}")
    print(f"As list: {m1.tolist()}")
    print(f"Flattened: {m1.flatten()}")
    
    # Element Access
    print("\n3. Element Access")
    print("-" * 20)
    
    print(f"Element at (0,1): {m1[0, 1]}")
    print(f"Row 0: {m1.get_row(0)}")
    print(f"Column 1: {m1.get_col(1)}")
    
    # Mathematical Operations
    print("\n4. Mathematical Operations")
    print("-" * 20)
    
    m2 = jm.FloatMatrix([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    
    print(f"m1: {m1}")
    print(f"m2: {m2}")
    print(f"m1 + m2: {m1 + m2}")
    print(f"m1 * m2: {m1 * m2}")
    print(f"Transpose of m1: {m1.T}")
    
    # Square matrices for matrix multiplication
    square1 = jm.FloatMatrix([[1.0, 2.0], [3.0, 4.0]])
    square2 = jm.FloatMatrix([[5.0, 6.0], [7.0, 8.0]])
    
    print(f"\nSquare matrix multiplication:")
    print(f"square1: {square1}")
    print(f"square2: {square2}")
    print(f"square1 @ square2: {square1 @ square2}")
    
    # Aggregation Functions
    print("\n5. Aggregation Functions")
    print("-" * 20)
    
    test_matrix = jm.FloatMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    print(f"Test matrix: {test_matrix}")
    print(f"Sum (total): {test_matrix.sum()}")
    print(f"Sum (axis=0, column sums): {test_matrix.sum(axis=0)}")
    print(f"Sum (axis=1, row sums): {test_matrix.sum(axis=1)}")
    print(f"Max: {test_matrix.max()}")
    print(f"Min: {test_matrix.min()}")
    
    # Statistical Functions
    print("\n6. Statistical Functions")
    print("-" * 20)
    
    print(f"Mean: {test_matrix.mean()}")
    print(f"Standard deviation: {test_matrix.std()}")
    print(f"Variance: {test_matrix.var()}")
    
    # Broadcasting
    print("\n7. Broadcasting")
    print("-" * 20)
    
    small = jm.FloatMatrix([[1.0, 2.0]])
    print(f"Small matrix: {small}")
    print(f"Broadcasted to (3,2): {small.broadcast((3, 2))}")
    
    # Matrix Inversion
    print("\n8. Matrix Inversion")
    print("-" * 20)
    
    try:
        invertible = jm.FloatMatrix([[2.0, 1.0], [1.0, 1.0]])
        print(f"Matrix: {invertible}")
        print(f"Inverse: {invertible.invert()}")
        
        # Verify A * A^-1 ≈ I
        identity_check = invertible @ invertible.invert()
        print(f"A @ A^-1: {identity_check}")
    except Exception as e:
        print(f"Inversion failed: {e}")
    
    print("\n" + "=" * 40)
    print("Demo completed!")

if __name__ == "__main__":
    main()