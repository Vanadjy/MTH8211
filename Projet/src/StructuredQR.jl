export isoverdetermined

using LinearAlgebra, Printf, SparseArrays, BlockDiagonals, BlockArrays

function isoverdetermined(A::Union{AbstractMatrix, AbstractBlockMatrix{T}}) where T
    # Ensures the input matrix A is overdetermined
    m, n = size(A)
    if m < n
        return error("Argument error : the input Matrix is not overdetermined")
    end
end

include("QOperations.jl")
include("QRBlocDiag.jl")
include("QRDense.jl")
include("QRhcat.jl")
include("qrhat.jl")
