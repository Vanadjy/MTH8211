using BenchmarkTools, Test

include("HouseholderCompact.jl")

"L'idée du code qui va suivre est de créer une factorisation QR d'une matrice A qui a été concaténnée 
horizontalement : ici, A = [A1 A2] où (m,n1 = size(A1)) et (m,n2 = size(A2))

Pour faire ça, nous allons avoir besoin de la factorisation QR de A1 dans un premier temps on aurait
alors A1 = Q1*[R1 ; 0], puis celle de (Q1_orth)*A2, d'où : (Q1_orth)*A2 = Q'[R' ; 0]
"

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

#On peut déjà commencer à comparer QR_concat_h! avec Householder_Compact!

#= m = 6
n1 = 2
n2 = 3
A1 = rand(m,n1)
A2 = rand(m,n2)
A = [A1 A2]

@benchmark QR_concat_h!($A, $n1)
@benchmark Householder_Compact!($A)
@benchmark QRHouseholder($A) =#

"A la vue des différents Benchmarks, QR_concat_h! n'améliore pas du tout ni le temps d'exécution, ni 
le nombre d'allocations faites ni l'espace mémoire occupé... Il nous faut donc une meilleure version
de cette fonction qui alloue moins de mémoire !

Au premier coup d'oeil : à chaque fois que l'on reconstruit la matrice Q, on peut gagner du temps et
de l'espace mémoire"

## Version compacte ##

"Pour cette version compacte, nous allons commencer par réaliser cette factorisation QR en évitant de 
former de nouveau les matrices Q dont on peut donner les impacts et les transformations en utilisant
les vecteurs u_j qui définissent ces matrices Q.

On a déjà une fonction qui donne l'opération Q*b avec b un vecteur, mais pas de produit matriciel, nous
allons donc déjà créer une version compacte du produit matriciel faisant intervenir Q :"

function mult_Q_B!(A, B)
    m, n = size(B)
    for j = 1:n
        B[1:end, j] = mult_Q_x!(A, B[1:end, j]) #produit vecteur colonne par vecteur colonne de B
    end
    B
end

#=m = 10
n = 8
A = rand(m, n)
B = copy(A)
B_aux = copy(B)
R = copy(A)
Householder_Compact!(R)
Q = Q_reconstruction!(R)
B_aux = mult_Q_B!(R, B_aux)
@test norm(Q*B - B_aux) <= 1e-14=#