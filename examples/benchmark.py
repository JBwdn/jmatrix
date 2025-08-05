"""
Performance benchmark for the jmatrix Python API.

This script demonstrates the performance characteristics of the jmatrix library
by running various operations on matrices of different sizes.
"""

import time
import jmatrix as jm


def benchmark_operation(name, func, *args, **kwargs):
    """Benchmark a single operation."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    duration = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"{name:25s}: {duration:8.3f} ms")
    return result


def main():
    print("JMatrix Performance Benchmark")
    print("=" * 50)
    
    # Small matrices (suitable for quick testing)
    print("\nSmall Matrices (100x100)")
    print("-" * 30)
    
    # Create test data
    data_small = [[float(i * 100 + j) for j in range(100)] for i in range(100)]
    
    m1 = benchmark_operation("Matrix creation", jm.FloatMatrix, data_small)
    m2 = benchmark_operation("Copy operation", m1.copy)
    
    # Basic operations
    benchmark_operation("Element access", lambda: m1[50, 50])
    benchmark_operation("Get row", m1.get_row, 50)
    benchmark_operation("Get column", m1.get_col, 50)
    benchmark_operation("Transpose", m1.transpose)
    
    # Mathematical operations
    benchmark_operation("Element-wise add", lambda: m1 + m2)
    benchmark_operation("Element-wise mul", lambda: m1 * m2)
    benchmark_operation("Matrix multiply", lambda: m1 @ m2)
    
    # Aggregations
    benchmark_operation("Sum (total)", m1.sum)
    benchmark_operation("Sum (axis=0)", m1.sum, axis=0)
    benchmark_operation("Sum (axis=1)", m1.sum, axis=1)
    benchmark_operation("Max", m1.max)
    benchmark_operation("Min", m1.min)
    
    # Statistics
    benchmark_operation("Mean", m1.mean)
    benchmark_operation("Std dev", m1.std)
    
    # Utility operations
    benchmark_operation("To list", m1.tolist)
    benchmark_operation("Flatten", m1.flatten)
    
    print("\nMedium Matrices (500x500)")
    print("-" * 30)
    
    # Medium matrices for more substantial timing
    print("Creating larger test matrices...")
    data_medium = [[float(i * 500 + j) for j in range(500)] for i in range(500)]
    
    m3 = benchmark_operation("Matrix creation", jm.FloatMatrix, data_medium)
    m4 = benchmark_operation("Copy operation", m3.copy)
    
    # Key operations on larger matrices
    benchmark_operation("Matrix multiply", lambda: m3 @ m4)
    benchmark_operation("Transpose", m3.transpose)
    benchmark_operation("Sum (total)", m3.sum)
    benchmark_operation("Mean", m3.mean)
    
    print("\nConstructor Performance")
    print("-" * 30)
    
    # Constructor performance
    benchmark_operation("zeros(100,100)", jm.zeros, (100, 100))
    benchmark_operation("ones(100,100)", jm.ones, (100, 100))
    benchmark_operation("eye(100)", jm.eye, 100)
    
    benchmark_operation("zeros(500,500)", jm.zeros, (500, 500))
    benchmark_operation("ones(500,500)", jm.ones, (500, 500))
    benchmark_operation("eye(500)", jm.eye, 500)
    
    print("\n" + "=" * 50)
    print("Benchmark completed!")
    print("\nNote: Performance will vary based on system specifications.")
    print("The Rust implementation provides significant speed advantages")
    print("for larger matrices and compute-intensive operations.")


if __name__ == "__main__":
    main()