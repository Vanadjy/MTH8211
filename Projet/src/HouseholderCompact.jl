using LinearAlgebra, Printf

## Version naïve ##

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

function QRHouseholder(A)
    "La fonction suivnte calcule et stocke les matrices liées à  la factorisation Qr de Householder
    
    La fonction prend en entrée une matrice de taille mxn et renvoie une matrice triangulaire supérieur 
    de taille mxn et une matrice unitaire de taille mxm"
    m, n = size(A)
    Q = I
    if m==n #gérer différemment le cas de matrice carrées
        for j = 1:(n-1)
            e1 = zeros(m-j+1)
            e1[1] = 1
            v = A[j:m,j]
            u = v + my_sign(v[1])*norm(v)*e1 
            Hj = HouseholderReflection(u) #dimensions de Hj -> m-j+1,m-j+1
            Qj = [I zeros(j-1,m-j+1) ;
                zeros(m-j+1,j-1) Hj]
            A = Qj*A
            Q = Q*Qj
        end
    else
        for j = 1:n
            e1 = zeros(m-j+1)
            e1[1] = 1
            v = A[j:m,j]
            u = v + my_sign(v[1])*norm(v)*e1 
            Hj = HouseholderReflection(u) #dimensions de Hj -> m-j+1,m-j+1
            Qj = [I zeros(j-1,m-j+1) ;
                zeros(m-j+1,j-1) Hj]
            A = Qj*A
            Q = Q*Qj
        end
    end
    Q, A
end

## Version Compacte ##

#Amélioration : appeler vj avant la boucle et sélectionner les valeurs qui nous intéressent plutôt que de l'appeler à chaque itération

## Version obsolète... ##
function Householder_Compact!(A)
    m, n = size(A)
    if m==n
        for j = 1:(n-1)
            vj = A[j:m,j]
            σj = my_sign(vj[1])
            vj_norm = norm(vj)
            vj[1] += σj*vj_norm
            vj ./= vj[1]
            δj = vj'vj

            #applicating Householder reflection
            A[j:m,j:n] .-= 2*vj*(vj'view(A,j:m,j:n))/δj
            
            #stock u_j
            A[j+1:m,j] .= vj[2:end]
        end
    else
        for j = 1:n
            vj = A[j:m,j]
            σj = my_sign(vj[1])
            vj_norm = norm(vj)
            vj[1] += σj*vj_norm
            vj ./= vj[1]
            δj = vj'vj
    
            #applicating Householder reflection
            A[j:m,j:n] -= 2*vj*(vj'A[j:m,j:n])/δj
            
            #stock u_j
            if j+1 <= m #prevent from being out of bounds for square or under-determined matrices
                A[j+1:m,j] = vj[2:end] 
            end
        end
    end
    A
end

function Householder_Compact_v2!(A)
    m, n = size(A)
    j = 1
        while (j <= n) & (j < m)
            vj = view(A,j+1:m,j)
            Aⱼⱼ = A[j,j]
            σj = my_sign(Aⱼⱼ)
            vj_norm = sqrt(Aⱼⱼ^2 + norm(vj)^2)
            vj ./= (Aⱼⱼ + σj*vj_norm)
            δj = vj'vj + 1

            #changing the diagonal term
            A[j,j] = -σj*vj_norm

            #applying Householder reflection
            uⱼ = view(vcat(1,vj),:,1)
            A[j:m,j+1:n] .-= 2*view(uⱼ,:,1).*(view(uⱼ,:,1)'view(A,j:m,j+1:n))./δj

            #going to next step
            j += 1
        end
    A
end

function Householder_Compact_v3!(A)
    m, n = size(A)
    j = 1
        while (j <= n) & (j < m)
            vj = view(A,j+1:m,j)
            Aⱼⱼ = A[j,j]
            σj = my_sign(Aⱼⱼ)
            vj_norm = sqrt(Aⱼⱼ^2 + norm(vj)^2)
            vj ./= (Aⱼⱼ + σj*vj_norm)
            δj = vj'vj + 1

            #changing the diagonal term
            A[j,j] = -σj*vj_norm

            #applying Householder on the jᵗʰ column (A[j,j+1:n])
            #B = A[j+1:m,j+1:n]
            @views A[j,j+1:n] .= (1-2/δj)*view(A,j,j+1:n) .- 2*(view(vj,:,1)'view(A,j+1:m,j+1:n))'/δj
            #println(norm(B-A[j+1:m,j+1:n])) #la sous-matrice A[j+1:m,j+1:n] n'est pas modifiée par cette ligne
            
            #applying Householder reflection (A[j+1:m, j+1:n])
            @views A[j+1:m,j+1:n] .-= 2*view(vj,:,1).*(view(A,j,j+1:n)' + (view(vj,:,1)'view(A,j+1:m,j+1:n)))/δj

            #going to next step
            j += 1
        end
    A
end



function Q_reconstruction!(A; Q=I)
    m, n = size(A)
    if m==n
        for j = 1:(n-1)
            u_j = [1 ; A[j+1:end,j]]
            Hj = HouseholderReflection(u_j)
            Qj = [I zeros(j-1,m-j+1) ;
                zeros(m-j+1,j-1) Hj]
            Q = Q*Qj
        end
    else
        for j = 1:n
            u_j = [1 ; A[j+1:end,j]]
            Hj = HouseholderReflection(u_j)
            Qj = [I zeros(j-1,m-j+1) ;
                zeros(m-j+1,j-1) Hj]
            Q = Q*Qj
        end
    end
    Q
end

function mult_Q_transpose_x!(A,x)
    m, n = size(A)
    if m==n
        for j = 1:(n-1)
            uj = [1 ; A[j+1:end,j]]
            δj = uj'uj
            x[j:m] -= 2*uj*(uj'x[j:m])/δj
        end
    else
        for j = 1:n
            uj = [1 ; A[j+1:end,j]]
            δj = uj'uj
            x[j:m] -= 2*uj*(uj'x[j:m])/δj
        end
    end
    x
end

function mult_Q_x!(A,x)
    m, n = size(A)
    if m==n
        for k = 2:n
            j = n-k+1
            uj = [1 ; A[j+1:end,j]]
            δj = uj'uj
            x[j:m] -= 2*uj*(uj'x[j:m])/δj
        end
    else
        for k = 1:n
            j = n-k+1
            uj = [1 ; A[j+1:end,j]]
            δj = uj'uj
            x[j:m] -= 2*uj*(uj'x[j:m])/δj
        end
    end
    x
end

m, n = 10, 8
A = rand(m,n)
b = rand(m)
R_H = copy(A)
Householder_Compact_v3!(R_H)
F = qr(A)
    
Q_H = Q_reconstruction!(R_H)
F.R - triu(R_H)[1:n,1:n]
F.Q - Q_H