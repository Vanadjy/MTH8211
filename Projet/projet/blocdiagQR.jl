#]add BenchmarkTools, UnicodePlots

]add NBInclude

using NBInclude
nbexport("Notebook.jl", "Notebook.ipynb")

jupytext --to jl blocdiagQRTest.ipynb

using LinearAlgebra
using Test
using BenchmarkTools
using UnicodePlots
using SparseArrays


include("HouseholderCompact.jl")

#   Les fonctions pour les matrices bloc-diagonales
#   –––––––––––––––––––––––––––––––––––––––––––––––––

# Calculate the QR factorization of a bloc-diagonale matrix A_vect. It return 2 matrix Q and R where Qk is 
# a mxm square matrix Qk= [Q | Q^perp] and Rk is an nxn triangular superior matrix without the zeros.
function blocdiagQR(A_vect, n_vect)
    Q = []
    Qort = []
    R = []   
    for k in eachindex(A_vect)
        Q1, R1 = qr(A_vect[k])
        push!(Q, Q1[:,1:n_vect[k]])
        push!(Qort, Q1[:,n_vect[k]+1:end])
        push!(R, R1)
    end
    Q = vcat([Q, Qort]...)
    return Q, R
end

# Calculate the QR factorization of a bloc-diagonale matrix A_vect. It return 2 matrix Q and R where Qk is 
# a mxm square matrix Qk= [Q | Q^perp] and Rk is an nxn triangular superior matrix without the zeros.
function blocdiagHQR(A_vect, n_vect)
    Q = []
    Qort = []
    R = []   
    for k in eachindex(A_vect)
        R1 = Householder_Compact!(A_vect[k])
        Q1 = Q_reconstruction!(R1)
        R1 = triu(R1[1:n_vect[k],:])
        push!(Q, Q1[:,1:n_vect[k]])
        push!(Qort, Q1[:,n_vect[k]+1:end])
        push!(R, R1)
    end
    Q = vcat([Q, Qort]...)
    return Q, R
end

# Calculate the QR factorization of a bloc-diagonale matrix A_vect. It return matrix A containing Q and R
function blocdiagHQR_compact!(A_vect, n_vect)
    for k in eachindex(A_vect)
        Householder_Compact!(A_vect[k])
    end
    return A_vect
end

#   Fonctions pour créer des matrices bloc-diagonales ou pour les
#  transformer de format vecteur à pleine.
#   –––––––––––––––––––––––––––––––––––––––––

# Create a block-diag A_vect random from a nb of submatrix and maximum of column and row of each submatrix. 
# Cannot be square matrix.
function CreateDiagBlock(nb_Matrix, m_max, n_max)
    A_vect = []
    m_vect = rand(1:m_max, nb_Matrix)
    n_vect = rand(1:n_max, nb_Matrix)
    for k = 1:nb_Matrix
        if m_vect[k]<n_vect[k]
            (m_vect[k],n_vect[k]) = (n_vect[k],m_vect[k])
        elseif m_vect[k] == n_vect[k]
            m_vect[k] += 1
        end
        push!(A_vect, rand(m_vect[k],n_vect[k]))
    end
    return A_vect, m_vect, n_vect
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
    return A_full, m_vect, n_vect
end

# Transform a block-diag A_vect to a block-diag A full
function VectToFull(A_vect, m_vect, n_vect)
    nb_Matrix = size(A_vect,1)
    A_full = zeros(sum(m_vect), sum(n_vect))
    
    for k = 1:nb_Matrix
        if k == 1
            A_full[1:m_vect[k],1:n_vect[k]] = A_vect[k]
        else
            A_full[sum(m_vect[1:k-1])+1:sum(m_vect[1:k]),sum(n_vect[1:k-1])+1:sum(n_vect[1:k])] = A_vect[k]
        end
    end
    return A_full
end

#   Test sur les différentes fonctions
#   ––––––––––––––––––––––––––––––––––––

# Test sur la fonction blocdiagQR

    A_vect, m_vect, n_vect = CreateDiagBlock(4, 4, 4)
    A_full = VectToFull(A_vect, m_vect, n_vect)
    
    Q_vect, R_vect = blocdiagQR(A_vect, n_vect)
    Qgood_vect = Q_vect[1:length(n_vect)]
    Qort_vect = Q_vect[length(n_vect)+1:end]
    Qgood_full = VectToFull(Qgood_vect, m_vect, n_vect)
    Qort_full = VectToFull(Qort_vect, m_vect, m_vect-n_vect)
    Q_full = hcat([Qgood_full, Qort_full]...)
    R_full = VectToFull(R_vect, n_vect, n_vect)
    
    m = sum(m_vect)
    n = sum(n_vect)
    R_zero = vcat(R_full, zeros(m-n,n))
    b = rand(m)

    F_full = qr(A_full)

@testset begin    
    @test norm(Q_full' * Q_full - I) <= 1e-14 #tests that Q_full is unitary.
    @test norm(Q_full * Q_full' - I) <= 1e-14 #tests that Q_full is unitary.
    @test norm(Qgood_full' * Qgood_full - I) <= 1e-14 #tests that Qgood_full is orthogonal.
    @test norm(Qgood_full*R_full - A_full) <= 1e-14 #tests if the QR decomposition is correct.
end


# Test sur la fonction blocdiagHQR
A_vect, m_vect, n_vect = CreateDiagBlock(3, 4, 4)
    A_full = VectToFull(A_vect, m_vect, n_vect)
    
    Q_vect, R_vect = blocdiagHQR(A_vect, n_vect)
    Qgood_vect = Q_vect[1:length(n_vect)]
    Qort_vect = Q_vect[length(n_vect)+1:end]
    Qgood_full = VectToFull(Qgood_vect, m_vect, n_vect)
    Qort_full = VectToFull(Qort_vect, m_vect, m_vect-n_vect)
    Q_full = hcat([Qgood_full, Qort_full]...)
    R_full = VectToFull(R_vect, n_vect, n_vect)

    m = sum(m_vect)
    n = sum(n_vect)
    R_zero = vcat(R_full, zeros(m-n,n))
    b = rand(m)

    F_full = qr(A_full);

@testset begin
    
    
    @test norm(Q_full' * Q_full - I) <= 1e-14 #tests that Q_full is unitary.
    @test norm(Q_full * Q_full' - I) <= 1e-14 #tests that Q_full is unitary.
    @test norm(Qgood_full' * Qgood_full - I) <= 1e-14 #tests that Qgood_full is orthogonal.
    @test norm(Qgood_full*R_full - A_full) <= 1e-14 #tests if the QR decomposition is correct.
end


# Test sur la fonction blocdiagHQR_compact!
A_vect, m_vect, n_vect = CreateDiagBlock(3, 4, 4)
A_full = VectToFull(A_vect, m_vect, n_vect)
    
R_vect = blocdiagHQR_compact!(A_vect, n_vect)

Qgood_vect = []
Qort_vect = []  
for k in eachindex(A_vect)
    R1 = R_vect[k]
    Q1 = Q_reconstruction!(R1)
    R1 = triu(R1[1:n_vect[k],:])
    push!(Qgood_vect, Q1[:,1:n_vect[k]])
    push!(Qort_vect, Q1[:,n_vect[k]+1:end])
    R_vect[k] = R1
end

Qgood_full = VectToFull(Qgood_vect, m_vect, n_vect)
Qort_full = VectToFull(Qort_vect, m_vect, m_vect-n_vect)
Q_full = hcat([Qgood_full, Qort_full]...)
R_full = VectToFull(R_vect, n_vect, n_vect)

m = sum(m_vect)
n = sum(n_vect)
R_zero = vcat(R_full, zeros(m-n,n))
    
b = rand(m)
F_full = qr(A_full);

@testset begin
    
    
    @test norm(Q_full' * Q_full - I) <= 1e-14 #tests that Q_full is unitary.
    @test norm(Q_full * Q_full' - I) <= 1e-14 #tests that Q_full is unitary.
    @test norm(Qgood_full' * Qgood_full - I) <= 1e-14 #tests that Qgood_full is orthogonal.
    @test norm(Qgood_full*R_full - A_full) <= 1e-14 #tests if the QR decomposition is correct.
end
