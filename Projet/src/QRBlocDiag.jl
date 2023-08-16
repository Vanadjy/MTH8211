export QRblocdiag!, QRblocdiag


"""
QRblocdiag!(A_vect::AbstractVector)

Computes the QR factorization of an overdetermined Block-Diagonal matrix A through all of its full-rank overdetermined diagonal blocks A₁, A₂, ... Aᵣ as A = QR where Q = diag(Q₁, Q₂, ..., Qᵣ) and R = diag(R₁, R₂, ..., Rᵣ) such that :

A₁ = Q₁R₁
A₂ = Q₂R₂
...
Aᵣ = QᵣRᵣ

Each Aᵢ block stores the QR information just like qrH! does : the upper triangular stores the coefficients of Rᵢ and the other coefficients of Aᵢ store the vectors required to rebuild Qᵢ

Computes : A = QR

Where : 
    - Q is an unitary Block-Diagonal matrix
    - R is an upper triangular Block-Diagonal matrix

#### Input arguments

* `A_vect` : an AbstractVector of size r containing all the blocks of a Block-Diagonal matrix;

#### Output arguments

* `A_vect` : the same vector where all the matrices have been QR-factorized;
"""

function QRblocdiag!(A_vect::AbstractVector)
    for A in A_vect
        qrH!(A)
    end
end

"""
QRblocdiag(A_vect::AbstractVector)

Computes the QR factorization of an overdetermined Block-Diagonal matrix A through all of its full-rank overdetermined diagonal blocks A₁, A₂, ... Aᵣ as A = QR where Q = diag(Q₁, Q₂, ..., Qᵣ) and R = diag(R₁, R₂, ..., Rᵣ) such that :

A₁ = Q₁R₁
A₂ = Q₂R₂
...
Aᵣ = QᵣRᵣ

Each Aᵢ block stores the QR information just like qrH! does : the upper triangular stores the coefficients of Rᵢ and the other coefficients of Aᵢ store the vectors required to rebuild Qᵢ

Computes : A = QR

Where : 
    - Q is an unitary Block-Diagonal matrix
    - R is an upper triangular Block-Diagonal matrix

#### Input arguments

* `A_vect` : an AbstractVector of size r containing all the blocks of a Block-Diagonal matrix;

#### Output arguments

* `B_vect` : a newly created AbstractVector of size r containing all the blocks of a Block-Diagonal matrix;
"""

function QRblocdiag(A_vect::AbstractVector)
    B_vect = copy(A_vect)
    QRblocdiag!(B_vect)
    B_vect
end

"""
QRblocdiag!(A::BlockDiagonal{T, Matrix{T}})

Computes the QR factorization of an overdetermined Block-Diagonal matrix A through all of its full-rank overdetermined diagonal blocks A₁, A₂, ... Aᵣ as A = QR where Q = diag(Q₁, Q₂, ..., Qᵣ) and R = diag(R₁, R₂, ..., Rᵣ) such that :

A₁ = Q₁R₁
A₂ = Q₂R₂
...
Aᵣ = QᵣRᵣ

Each Aᵢ block stores the QR information just like qrH! does : the upper triangular stores the coefficients of Rᵢ and the other coefficients of Aᵢ store the vectors required to rebuild Qᵢ

Computes : A = QR

Where : 
    - Q is an unitary Block-Diagonal matrix
    - R is an upper triangular Block-Diagonal matrix

#### Input arguments

* `A` : a BlockDiagonal{T, Matrix{T}} defined by its r blocks;

#### Output arguments

* `A` : the same BlockDiagonal Matrix whose r blocks now contain its own QR factorization information;
"""

function QRblocdiag!(A::BlockDiagonal{T, Matrix{T}}) where T
    j = 1
    N = nblocks(A)
    while j ≤ N
        qrH!(BlockDiagonals.blocks(A)[j])
        j+=1
    end
    A
end

"""
QRblocdiag(A::BlockDiagonal{T, Matrix{T}})

Computes the QR factorization of an overdetermined Block-Diagonal matrix A through all of its full-rank overdetermined diagonal blocks A₁, A₂, ... Aᵣ as A = QR where Q = diag(Q₁, Q₂, ..., Qᵣ) and R = diag(R₁, R₂, ..., Rᵣ) such that :

A₁ = Q₁R₁
A₂ = Q₂R₂
...
Aᵣ = QᵣRᵣ

Each Aᵢ block stores the QR information just like qrH! does : the upper triangular stores the coefficients of Rᵢ and the other coefficients of Aᵢ store the vectors required to rebuild Qᵢ

Computes : A = QR

Where : 
    - Q is an unitary Block-Diagonal matrix
    - R is an upper triangular Block-Diagonal matrix

#### Input arguments

* `A` : a BlockDiagonal{T, Matrix{T}} defined by its r blocks;

#### Output arguments

* `B` : a newly created BlockDiagonal Matrix whose r blocks now contain its own QR factorization information;
"""

function QRblocdiag(A::BlockDiagonal{T, Matrix{T}}) where T
    B = copy(A)
    QRblocdiag!(B)
    B
end