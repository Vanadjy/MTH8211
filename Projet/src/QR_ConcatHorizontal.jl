include("HouseholderCompact.jl")
include("blocdiagQR.jl")

"L'idée du code qui va suivre est de créer une factorisation QR d'une matrice A qui a été concaténnée 
horizontalement : ici, A = [A1 A2] où (m,n1 = size(A1)) et (m,n2 = size(A2))

Pour faire ça, nous allons avoir besoin de la factorisation QR de A1 dans un premier temps on aurait
alors A1 = Q1*[R1 ; 0], puis celle de (Q1_orth)*A2, d'où : (Q1_orth)*A2 = Q'[R' ; 0]
"

function splitH(A, n1)
    "La fonction suivante permet de séparer une matrice en deux blocs horizontaux :
    Le premier bloc possédant n₁ colonnes et le second (n - n₁)"

    return A[1:end, 1:n1], A[1:end, n1+1:end]
end

## Approche naïve ##

#Note : une structure pour gérer la forme particulière de A pourrait être pertinent

function QR_concat_h!(A, n1)
    "La fonction suivante calcule la factorisation QR d'une matrice concaténnée horizontalement,
    il est donc important que, dans le code qui suit, la matrice mise en entrée respecte bien cette 
    condition.

    La fonction prend en entrée une matrice A de taille mxn et un indice n1 qui indique la séparation
    des deux blocs de A.
    "
    m, n = size(A)
    A1 = A[1:end, 1:n1]
    A2 = A[1:end, n1+1:end]

    R1 = Householder_Compact!(A1)
    Q1 = Q_reconstruction!(R1)
    R1 = triu(R1)

    Q1_A2 = (Q1')*A2
    Q1_orth_A2 = Q1_A2[n1+1:end,1:end]
    R_prime = Householder_Compact!(Q1_orth_A2)
    Q_prime = Q_reconstruction!(R_prime)
    R_prime = triu(R_prime)

    Q = Q1*[I zeros(n1, m-n1); zeros(m-n1, n1) Q_prime]

    R = [R1 [Q1_A2[1:n1, 1:end] ; R_prime]]
    return Q, R
end

## Version compacte ##

"Pour cette version compacte, nous allons commencer par réaliser cette factorisation QR en évitant de 
former de nouveau les matrices Q dont on peut donner les impacts et les transformations en utilisant
les vecteurs u_j qui définissent ces matrices Q.

On a déjà une fonction qui donne l'opération Q*b avec b un vecteur, mais pas de produit matriciel, nous
allons donc déjà créer une version compacte du produit matriciel faisant intervenir Q :"

function mult_Q_B!(Q, B)
    n = size(B, 2)
    for j = 1:n
        #on effectue ici à chaque fois le produit compact Q*bⱼ
        B[1:end, j] = mult_Q_x!(Q, B[1:end, j]) #produit vecteur colonne par vecteur colonne de B
    end
    B
end

function mult_Q_transpose_B!(Q, B)
    n = size(B, 2)
    for j = 1:n
        #on effectue ici à chaque fois le produit compact Q*bⱼ
        B[1:end, j] = mult_Q_transpose_x!(Q, B[1:end, j]) #produit vecteur colonne par vecteur colonne de B
    end
    B
end

"Maintenant que nous avons nos produits matriciels compacts, nous pouvons maintenant nous attaquer 
à la version compacte de la factorisation QR avec concaténation horizontale"

function QR_concat_h_compact!(A1, A2)
    n1 = size(A1,2) #nombre de colonne du premier bloc
    m2, n2 = size(A2)

    Householder_Compact!(A1) #compute QR Householder on the first block of A : A1

    mult_Q_transpose_B!(A1, A2) #compute Q' *A2

    Q1_orth_A2 = view(A2,n1+1:m2,1:n2) #Q1_orth_A2 = A[n1+1:end, n1+1:end]
    Householder_Compact!(Q1_orth_A2) #compute QR Householder on Q1_orth_A2 which is the second block of Q1*A2

    return A1, A2
end

function Q_reconstruction_concatH(A1, A2) #A1 et A2 sont les blocs modifiés dans QR_concat_h_compact!
    m, n1 = size(A1)
    Q1 = Q_reconstruction!(A1)
    Q_prime = Q_reconstruction!(A2[n1+1:end, 1:end])
    return Q1*[I zeros(n1, m-n1); zeros(m-n1, n1) Q_prime]
end

function R_reconstruction_concatH(A1, A2)
    n1 = size(A1, 2)
    R1 = triu(A1)
    R_prime = triu(A2[n1+1:end, 1:end])
    R2 = [A2[1:n1, 1:end] ; R_prime]
    return [R1 R2]
end