export my_sign, HouseholderReflection, qrH!, QRebuild!, qtprod!, qprod!

using LinearAlgebra, Printf, SparseArrays

## utils ##

function my_sign(x::Number)
    if x == 0.0
        return 1
    else return sign(x)
    end
end


function HouseholderReflection(u::AbstractVector)
    "La fonction suivante calcule la réflexion de Householder associée au vecteur u 

    La fonction prend en entrée un AbstractVector et renvoie une matrice de dimensions mxm
    où m est la longueur du vecteur d'entrée u
    "
    δ = u'u
    return (I - 2*u*u'/δ)
end

function qrH!(A::AbstractMatrix)
    """
    qrH!(A)
    
    Computes the Householder QR factorization so that no additional memory space is allocated beyond that already occupied by the full rank input matrix A.
    
    Computes : A = QR
    
    Where :
        - Q is an unitary matrix (it means Q*Q = I)
        - R is upper triangular
    
    The strategy used here is first to overwrite the coefficients of R in the upper triangle of A. Besides, Q is defined only by vⱼ = A[j:m,j] (i.e. the elements of the jᵗʰ column of A beneath the diagonal), but the issue is only j-m elements can still be stored within A. That is why, those A[j:m,j] are scaled so that vⱼ[1] = 1. As we now know this information, there are j-m elements remaining in vⱼ that can be stored in A.

    #### Input arguments

    * `A`: a full rank matrix of dimension m × n;

    #### Output arguments

    * `A`: a matrix of dimension m × n containing the coefficients of Q and R;
    """
    m, n = size(A)
    j = 1
        while (j <= n) & (j < m)
            #necessary quantities
            vj = view(A,j:m,j)
            Aⱼⱼ = vj[1]
            σj = my_sign(Aⱼⱼ)
            vj_norm = norm(vj)
            vj[1] += σj*vj_norm
            δj = vj'vj

            #applying Householder reflection
            for l=j:n
                β = (vj'view(A,j:m,l))
                β *= (2/δj)
                for k=j:m
                    A[k,l] -= β*A[k,j]
                end
            end
            #scaling vj
            vj ./= vj[1]
            #changing the diagonal term
            A[j,j] = -σj*vj_norm
            #going to next step
            j += 1
        end
    A
end


function QRebuild!(A::AbstractMatrix; Q=I)
    """
    QRebuild!(A; Q=I)

    Rebuilds the unitary matrix Q from the information stored in the matrix A that has been transformed by the function Householder_Compact! so that A = QR
        
    Computes : Q so that Q is unitary (i.e. Q*Q = I)

    As the vectors stored in A are scaled, we can just get build the vector [1 ; A[j+1:end,j]] since, as it has been explained in the doc string of the function Householder_Compact!, the first element is supposed to be 1 and the others are scaled with respect to this. 

    We build the matrix Hⱼ which is the Reflexion of Householder of dimension (m-j+1)x(m-j+1) and Q is computed one step after the other according to the following rule :

    Q ← Q × Qⱼ where Qⱼ = [Iⱼ 0]
                          [0 Hⱼ]

    #### Input arguments

    * `A`: a full rank matrix of dimension m × n;

    #### Keyword arguments

    * `Q=I`: the identity matrix of dimension m × m , from which the rebuilding begins;
                      
    #### Output arguments

    * `Q`: the unitary matrix that fits with the QR decomposition of A

    NB : the purpose of this function is essentially to carry out tests to check if our QR decomposition verifies its uniqueness property (depending on the sign of the diagonal terms of R)
    """
    m, n = size(A)
    j = 1
    while (j <= n) & (j < m)
        u_j = [1 ; A[j+1:end,j]]
        Hj = HouseholderReflection(u_j)
        Qj = [I zeros(j-1,m-j+1) ;
            zeros(m-j+1,j-1) Hj]
        Q = Q*Qj
        j += 1
    end
    Q
end

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
end

function qprod!(A::AbstractMatrix,x::AbstractVector)
    m, n = size(A)
    k = 1
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
end