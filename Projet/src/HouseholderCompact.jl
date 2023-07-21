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


function Householder_Compact!(A)
    m, n = size(A)
    j = 1
        while (j <= n) & (j < m)
            vj = view(A,j:m,j)
            Aⱼⱼ = vj[1]
            σj = my_sign(Aⱼⱼ)
            vj_norm = norm(vj)
            vj[1] += σj*vj_norm
            δj = vj'vj

            #applying Householder reflection
            for l=j:n #procéder à QR Householder colonne par colonne
                β = (vj'view(A,j:m,l))
                β *= (2/δj)
                for k=j:m
                    A[k,l] -= β*A[k,j]
                end
            end
            
            vj ./= vj[1]
            #changing diagonal terms
            A[j,j] = -σj*vj_norm

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
    j = 1
    while (j <= n) & (j < m)
        uj = view(A,j+1:m,j)
        δj = uj'uj + 1
        xⱼ = x[j]
        β = uj'view(x,j+1:m,1)
        x[j] -= 2*(xⱼ + β)/δj
        x[j+1:m] .-= 2*(xⱼ + β)*uj/δj
        j+=1
    end
    x
end

function mult_Q_x!(A,x)
    m, n = size(A)
    k = 1
    while (k <= n) & (k < m)
        j = n-k+1
        uj = view(A,j+1:m,j)
        δj = uj'uj + 1
        xⱼ = x[j]
        β = uj'view(x,j+1:m,1)
        x[j] -= 2*(xⱼ + β)/δj
        x[j+1:m] .-= 2*(xⱼ + β)*uj/δj
        k+=1
    end
    x
end

m, n = 10, 8
A = rand(m,n)
b = rand(m)
b1 = rand(m)
R_H = copy(A)
Householder_Compact!(R_H)
F = qr(A)

Q_H = Q_reconstruction!(R_H)

#println(norm((Q_H')*b - mult_Q_transpose_x!(R_H,b)))
#(Q_H)*b - mult_Q_x!(R_H,b)