using LinearAlgebra, Printf

function my_sign(x::Number)
    if x == 0.0
        return 1
    else return sign(x)
    end
end

## compact version
#Amélioration : appeler vj avant la boucle et sélectionner les valeurs qui nous intéressent plutôt que de l'appeler à chaque itération
init_time = time()
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
            A[j:m,j:n] -= 2*vj*(vj'A[j:m,j:n])/δj
            
            #stock u_j
            A[j+1:m,j] = vj[2:end]
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

## On reprend ici une partie du code naïf pour vérifier notre implémentation de HouseholderCompact ##

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

#tests
#@printf "This code takes %s seconds to be fully computed" execution_time


" Notes : très bonne précision pour des systèmes surdéterminés, mais pb sur le dernier terme 
diagonal pour des systèmes carrés

La matrice reconstruite Q_H est unitaire, mais grandement différente de la matrice Q renvoyée 
par la fonction qr ce qui pose soucis étant donné que la factorisation QR est censée être unique... "

"Gain sur tous les points de vue avec la version compacte !"


#=Idées pour gérer les SparseBlocs -> créer une nouvelle structure DiagonalBlockSparseMatrix 
qui sera composée d'informations sur où se trouvent les blocs
AVANTAGE : on pourra gérer comme on le souhaite notre structure creuse
INCONVENIENT : structure qui n'existe pas a priori et donc possiblement difficile pour convertir nos 
matrices dans ce format...

                                    -> utiliser une structure déjà existante dans SparseArrays.jl
faire avec pour gérer nos structures creuses
AVANTAGE : on se casse pas la tête à créer une nouvelle structure qui risquerait de ne pas fonctionner
INCONVENIENT : pas le choix de se plier aux formalismes déjà créés pour des strctures creuses =#

execution_time = time()
execution_time -= init_time
#@printf "The script HouseholderCompact_v2.jl took %s seconds to be executed" execution_time
