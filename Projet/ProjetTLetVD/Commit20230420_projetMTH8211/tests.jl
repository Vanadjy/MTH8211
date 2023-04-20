include("HouseholderCompact.jl")
include("QR_ConcatHorizontal.jl")

using Test

@testset begin
    m, n = 10, 10
    A = rand(m,n)
    b = rand(m)
    R_H = copy(A)
    Householder_Compact!(R_H)
    F = qr(A)
    
    Q_H = Q_reconstruction!(R_H)

    @test norm(Q_H'Q_H - I) <= 1e-14 #tests that Q_H si unitary
    @test norm((Q_H')*b - mult_Q_transpose_x!(R_H,b)) <= 1e-14 #tests if the multiplication is correct
    @test norm((Q_H)*b - mult_Q_x!(R_H,b)) <= 1e-14
    @test norm(F.Q - Q_H) <= 1e-14 #tests of unicity of QR decomposition
    @test norm(F.R - triu(R_H[1:n,1:n])) <= 1e-14
    @test norm(Q_H*triu(R_H) - A) <= 1e-14 #tests if the QR decomposition is correct
end


@testset begin
    m = 6
    n1 = 2
    n2 = 3
    n = n1 + n2
    #=A1 = rand(m,n1)
    A2 = rand(m,n2)
    A = [A1 A2]=#
    A = rand(m ,n)
    A1, A2 = splitH(A, n1)
    B = copy(A)
    F = qr(A)

    Q, R = QR_concat_h!(A, n1)
    @test norm(F.R - R[1:n1+n2,1:n1+n2]) <= 1e-14 #test of unicity of QR decomposition
    @test norm(F.Q - Q) <= 1e-14
    @test norm(Q*R - A) <= 1e-14 #tests if the decomposition is correct

    R_H = copy(A)
    Householder_Compact!(R_H)
    Q_H = Q_reconstruction!(R_H)
    @test norm(Q_H*B - mult_Q_B!(R_H, B)) <= 1e-14 #tests if the multiplication QB is correct
    @test norm(Q_H'B - mult_Q_transpose_B!(R_H, B)) <= 1e-14 #tests if the multiplication Q*B is correct

    QR_concat_h_compact!(A1, A2)
    R = R_reconstruction_concatH(A1, A2)
    Q = Q_reconstruction_concatH(A1, A2)
    @test norm(F.R - R[1:n, 1:n]) <= 1e-14 #test of unicity of QR decomposition
    @test norm(F.Q - Q) <= 1e-14
    @test norm(A - Q*R) <= 1e-14 #tests if the decomposition is correct
end
