export CreateDiagBlock, QRblocdiag!

using LinearAlgebra
using Test
using SparseArrays, BlockDiagonals

include("HouseholderCompact.jl")

# Create a block-diag A_vect random from a nb of submatrix and maximum of column and row of each submatrix.
function CreateDiagBlock(nb_Matrix::Int, m_max::Int, n_max::Int)
    A_vect = Matrix{Float64}[]
    m_vect = rand(1:m_max, nb_Matrix)
    n_vect = rand(1:n_max, nb_Matrix)
    for k = 1:nb_Matrix
        if m_vect[k]<n_vect[k] #ensures to always being in overdetermined cases
            (m_vect[k],n_vect[k]) = (n_vect[k],m_vect[k])
        end
        push!(A_vect, rand(m_vect[k], n_vect[k]))
    end
    m = sum(m_vect)
    n = sum(n_vect)
    return A_vect, m , n 
end

## Upgrade blocdiagQR ##

function QRblocdiag!(A_vect::AbstractVector)
    for A in A_vect
        qrH!(A)
    end
end

function QRebuildBD(A_vect::AbstractVector; Q_vect=Matrix{Float64}[])
    for A in A_vect
        push!(Q_vect,QRebuild!(A))
    end
    return BlockDiagonal(Q_vect)
end