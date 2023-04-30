include("HouseholderCompact.jl")
include("QR_ConcatHorizontal.jl")
include("blocdiagQR.jl")


#L'idée maintenant c'est de faire du blocdiagQR sur A1, du QRHouseholder sur le bloc A2 et de faire du
#QR_concat_h sur ces deux blocs que l'on va concaténer horizontalement

"Nous allons maintenant créer une fonction qui procède à une factorisation QR avec concaténation horizontale où le bloc A1 est diagonal par blocs (structure analogue à celle que l'on peut retrouver pour des matrices de problèmes d'ajustements de faisceaux)"

function QR_PseudoBundle!(A_vect, A2) #A_vect est un vecteur de matrices qui constitue les sous-matrices d'une matrice bloc-diagonal et A2 est dense

    Q_vect, R_vect = blocdiagQR(A_vect) #compute QR Householder on the first block of A : A1
    Q1 = VectToFull(Q_vect)[1]
    R1 = VectToFull(R_vect)[1]
    n1 = size(R1, 2) #nombre de colonne du premier bloc
    R1 = [R1 ; zeros(size(Q1,1) - size(R1,1), size(R1,2))]
    

    Q1_mult_A2 = Q1'A2 #compute Q' *A2

    Q1_orth_A2 = Q1_mult_A2[n1+1:end,1:end] 
    Householder_Compact!(Q1_orth_A2) #compute QR Householder on Q1_orth_A2 which is the second block of Q1*A2

    Q2 = Q_reconstruction!(Q1_orth_A2)
    R2 = triu(Q1_orth_A2)

    return Q1, R1, Q2, R2, Q1_mult_A2[1:n1, 1:end]
end

#=function QR_concat_h_compact!(A1_vect, A2)
    n1 = size(A1,2) #nombre de colonne du premier bloc 

    blocdiagQR(A1_vect) #compute QR Householder on the first block of A : A1

    mult_Q_transpose_B!(A1, A2) #compute Q' *A2

    Q1_orth_A2 = A2[n1+1:end,1:end] #Q1_orth_A2 = A[n1+1:end, n1+1:end]
    Householder_Compact!(Q1_orth_A2) #compute QR Householder on Q1_orth_A2 which is the second block of Q1*A2

    A2[n1+1:end,1:end] = Q1_orth_A2

    return A1, A2
end=#