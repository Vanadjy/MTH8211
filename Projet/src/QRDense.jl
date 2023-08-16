export my_sign, qrH!, qrH

"""
    my_sign(x::Number)

    The idea of this function is to extend the sign one such that it returns 1 whenever it is evaluated at 0

    #### Input arguments

    * `x` : a number

    #### Output arguments

    * 1 if x==0.0
    * the sign of x otherwise
    """
function my_sign(x::Number)
    return x == 0 ? 1 : sign(x)
end

"""
qrH!(A::AbstractMatrix)

Computes the Householder QR factorization so that no additional memory space is allocated beyond that already occupied by the full rank overdetermined input matrix A.

Computes : A = QR

Where :
    - Q is an unitary matrix (it means Q*Q = I)
    - R is upper triangular

The strategy used here is first to overwrite the coefficients of R in the upper triangle of A. Besides, Q is defined only by vⱼ = A[j:m,j] (i.e. the elements of the jᵗʰ column of A beneath the diagonal), but the issue is only j-m elements can still be stored within A. That is why, those A[j:m,j] are scaled so that vⱼ[1] = 1. As we now know this information, there are j-m elements remaining in vⱼ that can be stored in A.

#### Input arguments

* `A`: a full rank matrix of dimension m × n, m ≥ n;

#### Output arguments

* `A`: a matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
"""
function qrH!(A::AbstractMatrix)
    m, n = size(A)
    isoverdetermined(A)
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

"""
qrH(A::AbstractMatrix)

Computes the Householder QR factorization so that no additional memory space is allocated beyond that already occupied by the full rank overdetermined input matrix A.

Computes : A = QR

Where :
    - Q is an unitary matrix (it means Q*Q = I)
    - R is upper triangular

The strategy used here is first to overwrite the coefficients of R in the upper triangle of A. Besides, Q is defined only by vⱼ = A[j:m,j] (i.e. the elements of the jᵗʰ column of A beneath the diagonal), but the issue is only j-m elements can still be stored within A. That is why, those A[j:m,j] are scaled so that vⱼ[1] = 1. As we now know this information, there are j-m elements remaining in vⱼ that can be stored in A.

#### Input arguments

* `A`: a full rank matrix of dimension m × n, m ≥ n;

#### Output arguments

* `B`: a newly created matrix of dimension m × n, m ≥ n, containing the coefficients of Q and R;
"""
function qrH(A::AbstractMatrix)
    B = copy(A)
    qrH!(B)
    B
end