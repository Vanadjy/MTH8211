include("HouseholderCompact.jl")
include("QR_ConcatHorizontal.jl")
include("blocdiagQR.jl")
include("MatricesTestSet.jl")
include("StructuredQR.jl")

using Test

#tests sur des matrices carrées
@testset begin
    m, n = 10, 10
    A = rand(m,n)
    b1 = rand(m)
    b2 = rand(m)
    R_H = copy(A)
    Householder_Compact!(R_H)
    F = qr(A)
    
    Q_H = Q_reconstruction!(R_H)

    @test norm(Q_H'Q_H - I) <= 1e-13 #tests that Q_H si unitary
    @test norm((Q_H')*b1 - mult_Q_transpose_x!(R_H,b1)) <= 1e-13 #tests if the multiplication is correct
    #@test norm((Q_H)*b2 - mult_Q_x!(R_H,b2)) <= 1e-13
    @test norm(F.Q - Q_H) <= 1e-13 #tests of unicity of QR decomposition
    @test norm(F.R - triu(R_H[1:n,1:n])) <= 1e-13
    @test norm(Q_H*triu(R_H) - A) <= 1e-13 #tests if the QR decomposition is correct
end

#tests sur des matrices surdéterminées
@testset begin
    m, n = 10, 8
    A = rand(m,n)
    b = rand(m)
    R_H = copy(A)
    Householder_Compact!(R_H)
    F = qr(A)
    
    Q_H = Q_reconstruction!(R_H)

    @test norm(Q_H'Q_H - I) <= 1e-13 #tests that Q_H si unitary
    @test norm((Q_H')*b - mult_Q_transpose_x!(R_H,b)) <= 1e-13 #tests if the multiplication is correct
    @test norm((Q_H)*b - mult_Q_x!(R_H,b)) <= 1e-13
    @test norm(F.Q - Q_H) <= 1e-13 #tests of unicity of QR decomposition
    @test norm(F.R - triu(R_H[1:n,1:n])) <= 1e-13
    @test norm(Q_H*triu(R_H) - A) <= 1e-13 #tests if the QR decomposition is correct
end

#tests sur des matrices surdéterminées
@testset begin
    m = 6
    n1 = 2
    n2 = 3
    n = n1 + n2

    A = rand(m ,n)
    A1, A2 = splitH(A, n1)
    B = copy(A)
    F = qr(A)

    Q, R = QR_concat_h!(A, n1)
    @test norm(F.R - R[1:n1+n2,1:n1+n2]) <= 1e-13 #test of unicity of QR decomposition
    @test norm(F.Q - Q) <= 1e-13
    @test norm(Q*R - A) <= 1e-13 #tests if the decomposition is correct

    R_H = copy(A)
    Householder_Compact!(R_H)
    Q_H = Q_reconstruction!(R_H)
    @test norm(Q_H*B - mult_Q_B!(R_H, B)) <= 1e-13 #tests if the multiplication QB is correct
    @test norm(Q_H'B - mult_Q_transpose_B!(R_H, B)) <= 1e-13 #tests if the multiplication Q*B is correct

    QR_concat_h_compact!(A1, A2)
    R = R_reconstruction_concatH(A1, A2)
    Q = Q_reconstruction_concatH(A1, A2)
    @test norm(F.R - R[1:n, 1:n]) <= 1e-14 #test of unicity of QR decomposition
    @test norm(F.Q - Q) <= 1e-13
    @test norm(A - Q*R) <= 1e-13 #tests if the decomposition is correct
end

#tests sur des matrices carrées
@testset begin
    m = 6
    n1 = 3
    n2 = 3
    n = n1 + n2

    A = rand(m ,n)
    A1, A2 = splitH(A, n1)
    B = copy(A)
    F = qr(A)

    Q, R = QR_concat_h!(A, n1)
    @test norm(F.R - R[1:n1+n2,1:n1+n2]) <= 1e-14 #test of unicity of QR decomposition
    @test norm(F.Q - Q) <= 1e-13
    @test norm(Q*R - A) <= 1e-13 #tests if the decomposition is correct

    R_H = copy(A)
    Householder_Compact!(R_H)
    Q_H = Q_reconstruction!(R_H)
    #@test norm(Q_H*B - mult_Q_B!(R_H, B)) <= 1e-13 #tests if the multiplication QB is correct
    @test norm(Q_H'B - mult_Q_transpose_B!(R_H, B)) <= 1e-13 #tests if the multiplication Q*B is correct

    QR_concat_h_compact!(A1, A2)
    R = R_reconstruction_concatH(A1, A2)
    Q = Q_reconstruction_concatH(A1, A2)
    @test norm(F.R - R[1:n, 1:n]) <= 1e-13 #test of unicity of QR decomposition
    @test norm(F.Q - Q) <= 1e-13
    @test norm(A - Q*R) <= 1e-13 #tests if the decomposition is correct
end


@testset begin
    for k=1:50
        A1 = A_vect_collection[k]
        A1_full = VectToFull(A1)[1]
        A2 = B_collection[k]
        
        A = [A1_full A2]
        
        F = qr(A)
        F1 = qr(A1_full)
                
        Q1, R1, Q2, R2, Q1_mult_A2 = QR_PseudoBundle!(A1, A2)
        
        n1 = size(A1,2)
        
        k1 = size(Q1,1)
        k2 = size(Q2,1) #k1 ≥ k2
        
        Q = Q1*[I zeros(k1-k2, k2) ; zeros(k2, k1-k2) Q2]
        
        R = [R1  [Q1_mult_A2; R2]]
        
        @test norm(F.R - R[1:size(R,2), 1:size(R,2)]) ≤ 1e-13 #tests the unicity of the decomposition of A
        @test norm(F.Q - Q) ≤ 1e-13
        @test norm(A - Q*R) ≤ 1e-13 #tests if the QR decomposition of A is correct
    end
end

"NB : pour le moment, mult_Q_x! ne fonctionne pas pour des matrices carrées, à revoir"