export qprod!, qtprod!, qprod, qtprod, qmul!, qtmul!, qmul, qtmul

### DISCLAIMER ###
#=
Pour le moment, je considère que l'utilisateur a appliqué sa factorisation QR sur A et qu'il met en entrée sa matrice factorisée. Pour améliorer cela, il faudrait rajouter une structure QRFactorization qui donne entre autres l'information si la factorisation QR a été réalisée ou non (à l'avenir, cette structure deviendrait comme celle renvoyée par qr) 
=#


"""
qtprod!(A::AbstractMatrix,x::AbstractVector)

Calculates the product of Qᵀ, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by a vector x and stores the result within x by replacing its values by those of Qᵀx

Computes : Q*x

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - x is a vector

#### Input arguments

* `A`: an overtermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `x`: a vector of size m 

#### Output arguments

* `x`: a vector of size m containing the result of the operation Q*x;
"""

function qtprod!(A::AbstractMatrix,x::AbstractVector)
    m, n = size(A)
    j = 1
    while (j <= n) & (j < m)
        uj = view(A,j+1:m,j)
        δj = uj'uj + 1
        xⱼ = x[j]
        β = 0
        for i = 1:m-j
            β += uj[i]*x[i+j]
        end
        x[j] -= 2*(xⱼ + β)/δj
        for l = j+1:m
            x[l] -= 2*(xⱼ + β)*A[l,j]/δj
        end
        j+=1
    end
    x
end

"""
qprod!(A::AbstractMatrix,x::AbstractVector)

Calculates the product of Q, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by a vector x and stores the result within x by replacing its values by those of Qᵀx

Computes : Qx

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - x is a vector

#### Input arguments

* `A`: an overtermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `x`: a vector of size m 

#### Output arguments

* `x`: a vector of size m containing the result of the operation Qx;
"""
function qprod!(A::AbstractMatrix,x::AbstractVector)
    m, n = size(A)
    k = 1
    if m==n #condition to treat the square case
        k +=1
    end
    while (k <= n) & (k <= m)
        j = n-k+1
        uj = view(A,j+1:m,j)
        δj = uj'uj + 1
        xⱼ = x[j]
        β = 0
        for i = 1:m-j
            β += uj[i]*x[i+j]
        end
        x[j] -= 2*(xⱼ + β)/δj
        for l = j+1:m
            x[l] -= 2*(xⱼ + β)*A[l,j]/δj
        end
        k+=1
    end
    x
end

"""
qprod(A::AbstractMatrix,x::AbstractVector)

Calculates the product of Q, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by a vector x and stores the result within y, a new vector with the same size as x

Computes : Qx

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - x is a vector

#### Input arguments

* `A`: an overdetermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `x`: a vector of size m 

#### Output arguments

* `y`: a newly created vector of size m containing the result of the operation Qx;
"""

function qprod(A::AbstractMatrix,x::AbstractVector)
    y = copy(x)
    qprod!(A, y)
    y
end

"""
qtprod(A::AbstractMatrix,x::AbstractVector)

Calculates the product of Qᵀ, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by a vector x and stores the result within y, a new vector with the same size as x

Computes : Q*x

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - x is a vector

#### Input arguments

* `A`: an overdetermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `x`: a vector of size m 

#### Output arguments

* `y`: a newly created vector of size m containing the result of the operation Q*x;
"""

function qtprod(A::AbstractMatrix,x::AbstractVector)
    y = copy(x)
    qtprod!(A, y)
    y
end

"""
qmul!(A::AbstractMatrix, B::AbstractMatrix)

Calculates the multiplication of Q, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by an other matrix B and stores the result within it by replacing its values by those of QB

Computes : QB

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - B is a matrix

#### Input arguments

* `A`: an overdetermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `B`: a matrix of dimensions m × r

#### Output arguments

* `B`: a matrix of dimensions m × r containing the result of the operation QB;
"""

function qmul!(A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(B)
    for j = 1:n
        #we apply here qprod! on each column of B : Calculates Qbⱼ
        qprod!(A, view(B, 1:m, j))
    end
    B
end

"""
qmul!(A::AbstractMatrix, B::AbstractMatrix)

Calculates the multiplication of Q, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by an other matrix B and stores the result within Y by replacing its values by those of QB

Computes : QB

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - B is a matrix

#### Input arguments

* `A`: an overdetermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `B`: a matrix of dimensions m × r

#### Output arguments

* `Y`: a newmy created matrix of dimensions m × r containing the result of the operation QB;
"""

function qmul(A::AbstractMatrix, B::AbstractMatrix)
    Y = copy(B)
    qmul!(A, Y)
    Y
end

"""
qtmul!(A::AbstractMatrix, B::AbstractMatrix)

Calculates the multiplication of Qᵀ, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by an other matrix B and stores the result within it by replacing its values by those of QᵀB

Computes : QᵀB

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - B is a matrix

#### Input arguments

* `A`: an overdetermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `B`: a matrix of dimensions m × r

#### Output arguments

* `B`: a matrix of dimensions m × r containing the result of the operation QᵀB;
"""

function qtmul!(A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(B)
    for j = 1:n
        #we apply here qtprod! on each column of B : Calculates Qᵀbⱼ
        qtprod!(A, view(B, 1:m, j))
    end
    B
end

"""
qtmul!(A::AbstractMatrix, B::AbstractMatrix)

Calculates the multiplication of Qᵀ, the unitary matrix from the QR decomposition of A (an overdetermined full-rank matrix), by an other matrix B and stores the result within Y by replacing its values by those of QᵀB

Computes : QᵀB

Where :
    - Q is the unitary matrix from the QR factorization of A (an overdetermined full-rank matrix)
    - B is a matrix

#### Input arguments

* `A`: an overdetermined full rank matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
* `B`: a matrix of dimensions m × r

#### Output arguments

* `Y`: a newly created matrix of dimensions m × r containing the result of the operation QᵀB;
"""

function qtmul(A::AbstractMatrix, B::AbstractMatrix)
    Y = copy(B)
    qtmul!(A, Y)
    Y
end

function get_r(A::Union{AbstractMatrix, AbstractSparseMatrix, BlockDiagonal{T, Matrix{T}}, AbstractBlockMatrix{T}}) where T
    n = size(A, 2)
    return triu(view(A, 1:n, 1:n))
end

"""
    rdiv!()
"""
function rdiv!(A, x)
    
end

#=m = 10
n = 8
A = rand(m, n)
B = copy(A)
qrhat!(A)
F = qr(B)

F.R - get_r(A)=#