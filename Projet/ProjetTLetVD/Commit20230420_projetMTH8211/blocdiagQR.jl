using LinearAlgebra
using Test
using BenchmarkTools

include("HouseholderCompact.jl")

# Calculate the QR factorization of a bloc-diagonale matrix A_vect. It return a vector F containing Q_k and R_k matrix
# of each block.
function blocdiagQR(A_vect)
    Q = []
    R = []
    m_vect = []
    n_vect = []      
    
    for k = 1:size(A,1)
        push!(m_vect,size(A_vect[k],1))
        push!(n_vect,size(A_vect[k],2))
        Q1, R1 = qr(A_vect[k])
        push!(Q, Q1)
        push!(R, R1)
    end
    return Q, R
end

# Create a block-diag A_vect random from a nb of submatrix and maximum of column and row of each submatrix.
function CreateDiagBlock(nb_Matrix, m_max, n_max)
    A_vect = []
    m_vect = rand(1:m_max, nb_Matrix)
    n_vect = rand(1:n_max, nb_Matrix)
    for k = 1:nb_Matrix
        if m_vect[k]<n_vect[k] #ensures to always being in overdetermined cases
            (m_vect[k],n_vect[k]) = (n_vect[k],m_vect[k])
        end
        push!(A_vect, rand(m_vect[k],n_vect[k]))
    end
    m = sum(m_vect)
    n = sum(n_vect)
    return A_vect, m , n 
end

# Create a block-diag A random from a nb of submatrix and maximum of column and row of each submatrix.
function CreateDiagBlockFull(nb_Matrix, m_max, n_max)
    m_vect = rand(1:m_max, nb_Matrix)
    n_vect = rand(1:n_max, nb_Matrix)
    for k = 1:nb_Matrix
        if m_vect[k]<n_vect[k]
            (m_vect[k],n_vect[k]) = (n_vect[k],m_vect[k])
        end 
    end
    A_full = zeros(sum(m_vect), sum(n_vect))
            
    for k = 1:nb_Matrix
        if k == 1
            A_full[1:m_vect[k],1:n_vect[k]] = rand(m_vect[k],n_vect[k])
        else
            A_full[sum(m_vect[1:k-1])+1:sum(m_vect[1:k]),sum(n_vect[1:k-1])+1:sum(n_vect[1:k])] = rand(m_vect[k],n_vect[k])
        end
    end
    m = sum(m_vect)
    n = sum(n_vect)
        
    return A_full, m, n
end

# Transform a block-diag A_vect to a block-diag A full
function VectToFull(A_vect)
    nb_Matrix = size(A_vect,1)
    m_vect = []
    n_vect = []
    for k = 1:nb_Matrix
        push!(m_vect,size(A_vect[k],1))
        push!(n_vect,size(A_vect[k],2))
    end
    A_full = zeros(sum(m_vect), sum(n_vect))
    
    for k = 1:nb_Matrix
        if k == 1
            A_full[1:m_vect[k],1:n_vect[k]] = A_vect[k]
        else
            A_full[sum(m_vect[1:k-1])+1:sum(m_vect[1:k]),sum(n_vect[1:k-1])+1:sum(n_vect[1:k])] = A_vect[k]
        end
    end
    m = sum(m_vect)
    n = sum(n_vect)
    return A_full, m, n
end