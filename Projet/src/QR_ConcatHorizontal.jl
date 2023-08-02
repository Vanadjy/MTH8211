export splitH, qmul!, qtmul!, QRhcat!, QRebuildHcat, RRebuildHcat

include("HouseholderCompact.jl")
include("blocdiagQR.jl")

"L'idée du code qui va suivre est de créer une factorisation QR d'une matrice A qui a été concaténnée 
horizontalement : ici, A = [A1 A2] où (m,n1 = size(A1)) et (m,n2 = size(A2))

Pour faire ça, nous allons avoir besoin de la factorisation QR de A1 dans un premier temps on aurait
alors A1 = Q1*[R1 ; 0], puis celle de (Q1_orth)*A2, d'où : (Q1_orth)*A2 = Q'[R' ; 0]
"

function splitH(A::AbstractMatrix, n1::Int)
    "La fonction suivante permet de séparer une matrice en deux blocs horizontaux :
    Le premier bloc possédant n₁ colonnes et le second (n - n₁)"

    return A[1:end, 1:n1], A[1:end, n1+1:end]
end

## Version compacte ##

"Pour cette version compacte, nous allons commencer par réaliser cette factorisation QR en évitant de 
former de nouveau les matrices Q dont on peut donner les impacts et les transformations en utilisant
les vecteurs u_j qui définissent ces matrices Q.

On a déjà une fonction qui donne l'opération Q*b avec b un vecteur, mais pas de produit matriciel, nous
allons donc déjà créer une version compacte du produit matriciel faisant intervenir Q :"

function qmul!(A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(B)
    for j = 1:n
        #on effectue ici à chaque fois le produit compact Q*bⱼ
        qprod!(A, view(B, 1:m, j)) #produit vecteur colonne par vecteur colonne de B
    end
    B
end

function qtmul!(A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(B)
    for j = 1:n
        #on effectue ici à chaque fois le produit compact Q*bⱼ
        qtprod!(A, view(B, 1:m, j)) #produit vecteur colonne par vecteur colonne de B
    end
    B
end

"Maintenant que nous avons nos produits matriciels compacts, nous pouvons maintenant nous attaquer 
à la version compacte de la factorisation QR avec concaténation horizontale"

function QRhcat!(A1::AbstractMatrix, A2::AbstractMatrix)
    n1 = size(A1,2) #nombre de colonne du premier bloc
    m2, n2 = size(A2)

    qrH!(A1) #compute QR Householder on the first block of A : A1

    qtmul!(A1, A2) #compute Q' *A2

    Q1_orth_A2 = view(A2,n1+1:m2,1:n2) #Q1_orth_A2 = A[n1+1:end, n1+1:end]
    qrH!(Q1_orth_A2) #compute QR Householder on Q1_orth_A2 which is the second block of Q1*A2

    return A1, A2
end

function QRebuildHcat(A1::AbstractMatrix, A2::AbstractMatrix) #A1 et A2 sont les blocs modifiés dans QR_concat_h_compact!
    m, n1 = size(A1)
    Q1 = QRebuild!(A1)
    Q_prime = QRebuild!(A2[n1+1:end, 1:end])
    return Q1*[I zeros(n1, m-n1); zeros(m-n1, n1) Q_prime]
end

function RRebuildHcat(A1::AbstractMatrix, A2::AbstractMatrix)
    n1 = size(A1, 2)
    R1 = triu(A1)
    R_prime = triu(A2[n1+1:end, 1:end])
    R2 = [A2[1:n1, 1:end] ; R_prime]
    return [R1 R2]
end