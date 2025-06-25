

function matrix_multiply(rows::Int32, cols::Int32)
    # Define two matrices of size rows x cols
    mat1 = rand(rows, cols)
    mat2 = rand(rows, cols)

    # Multiply matrices
    result = mat1 * mat2
    return result
end

rows::Int32 = 20000
cols::Int32 = 20000

@time res = matrix_multiply(rows, cols)