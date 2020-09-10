---
title: "NN from Scratch, Part 1: Matrix Multiplication"
date: 2020-06-29T11:00:26.000Z
tags:
  - neural networks
  - matrices
  - julia
  - python
type: posts
---
## Introduction

This first series will attempt to explain the inner workings of neural networks. Its aim is to create a state-of-the-art neural network from scratch. It is directly inspired from part 2 of the 2019 edition of the [fastai course](https://course.fast.ai). All code snippets will be available [online](https://gitlab.com/de-souza/nn-from-scratch).

### What is a neural network?

A neural network is a way to approximate a function through a series of matrix multiplications. This function can be anything, from a simple linear function to a highly non-linear function -- such as whether a picture contains a cat.

### Vocabulary

A distinct vocabulary is used to describe the matrices forming a neural network. The first matrix, holding the input of the approximated function, is called _input layer_. Depending on their purpose, subsequent matrices are called _parameters_ or _activations_. Finally, the matrix which holds the prediction -- the result of the approximated function -- is called _output layer_.

## Matrix Multiplication

The most fundamental process in neural networks is matrix multiplication[](http://matrixmultiplication.xyz). The goal of this first post is to implement matrix multiplication from scratch.

To be able to multiply two matrices, the first matrix must have as many columns as there are rows in the second matrix. The result is a new matrix that shares its number of rows with first matrix and its number of columns with the second. Each element of the resulting matrix is then calculated as the sum of the [element-wise multiplication](http://matrixmultiplication.xyz) of a row of the first matrix with a column of the second matrix.

### Matrix Multiplication in Julia

In Julia, this algorithm can be implemented as follows.

```julia
function matmul(mat_a::AbstractMatrix, mat_b::AbstractMatrix)
    rows_a, columns_a = size(mat_a)
    rows_b, columns_b = size(mat_b)
    columns_a == rows_b || throw(DimensionMismatch("matrices must match"))
    mat_c = zeros(rows_a, columns_b)
    for k = 1:columns_a, j = 1:columns_b, i = 1:rows_a
        mat_c[i, j] += mat_a[i, k] * mat_b[k, j]
    end
    mat_c
end
```

This yields the same result as `matmul`, the built-in implementation for matrix multiplication in Julia. Let's compare their speed.

```julia
julia> using BenchmarkTools
julia> mat_a = rand(200, 300)
julia> mat_b = rand(300, 200)
julia> @btime matmul(mat_a, mat_b)
  17.215 ms (2 allocations: 312.58 KiB)
julia> @btime mat_a * mat_b
  685.237 μs (2 allocations: 312.58 KiB)
```

The built-in implementation is 25 times faster!

### Matrix Multiplication in Python

The same algorithm can be implemented in Python. To speed things up, the PyTorch library is used to create the matrices.

```python
import torch

def matmul(mat_a, mat_b):
    rows_a, columns_a = mat_a.shape
    rows_b, columns_b = mat_b.shape
    if columns_a != rows_b:
        raise ValueError("dimension mismatch")
    mat_c = torch.zeros(rows_a, columns_b)
    for i in range(rows_a):
        for j in range(columns_b):
            for k in range(columns_a):
                mat_c[i, j] += mat_a[i, k] * mat_b[k, j]
    return mat_c
```

Due to the interpreted nature Python, this code already runs much slower than the same algorithm in Julia. Let's compare its speed with the implementation from PyTorch:

```python
>>> import timeit
>>> mat_a = torch.rand(10, 20)
>>> mat_b = torch.rand(20, 30)
>>> min(timeit.repeat(
... "matmul(mat_a, mat_b)", repeat=5, number=1, globals=globals()))
0.21406823999859625  # 214 ms
>>> min(timeit.repeat(
... "mat_a @ mat_b", repeat=5, number=1, globals=globals()))
1.712999801384285e-05  # 17.1 µs
```

The implementation from PyTorch is 10,000 times faster!

### Speed Increase

The matrix multipllication implementations found in Julia and the PyTorch library were much faster that this algorithm. Why?

The reason is that Julia and PyTorch directly pass matrix multiplications to highly-optimised, low-level subroutines called [basic linear algebra subprograms](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (BLAS).

To speed things up and now that it is clear how matrix multiplications work, the BLAS implementation of matrix multiplication will be used in the next posts.
