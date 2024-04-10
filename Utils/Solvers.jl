## Linear solvers
using LinearAlgebra

# Modified Thomas algorithm - 4 diagonals
function modifiedThomasAlgorithm(A, d)
    # Modify the 4-diagonal matrix A to be a 3-diagonal matrix
    n = length(d)
    A_ = copy(A)
    d_ = copy(d)

    for i = 3:n
        m = A_[i, i - 2] / A_[i - 1, i - 2]
        A_[i, i - 1] = A_[i, i - 1] - m * A_[i - 1, i - 1]
        A_[i, i] = A_[i, i] - m * A_[i - 1, i]
        try A_[i, i + 1] = A_[i, i + 1] - m * A_[i - 1, i + 1] catch end
        d_[i] = d_[i] - m * d_[i - 1]
    end

    # Thomas algorithm
    x = thomasAlgorithm(A_, d_)
    return x
end

# Thomas Algorithm - 3 diagonals
function thomasAlgorithm(A, d)
    a = diag(A, -1)
    b = diag(A, 0)
    c = diag(A, 1)

    d_ = copy(d)

    n = length(d)
    x = zeros(n)

    # Forward elimination
    for i = 2:n
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d_[i] = d_[i] - m * d_[i - 1]
    end

    # Backward substitution
    x[end] = d_[end] / b[end]
    for i = n-1:-1:1
        x[i] = (d_[i] - c[i] * x[i + 1]) / b[i]
    end

    return x
end

# # test
# A = [2. 2. 0. 0. 0.; 2. 3. 4. 0. 0.; 1. 4. 5. 2. 0.; 0. 1. 2. 3. 4.; 0. 0. 1. 2. 3.]
# d = [1., 2., 3., 4., 5.]

# x = modifiedThomasAlgorithm(A, d)
# x_e = A \ d
# println(x)
# println(x_e)
