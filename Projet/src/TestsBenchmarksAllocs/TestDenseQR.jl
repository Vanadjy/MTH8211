## Tests of dense Householder QR factorization

## Overdetermined matrices ##
@testset begin
    m, n = 10, 8
    A = rand(m,n)
    b = rand(m)
    b1 = deepcopy(b)
    b2 = deepcopy(b)
    R_H = deepcopy(A)
    qrH!(R_H)
    F = qr(A)
    
    Q_H = QRebuild!(R_H)

    @test norm(Q_H'Q_H - I) <= 1e-13 #tests that Q_H si unitary
    qtprod!(R_H,b1)
    @test norm((Q_H')*b - b1) <= 1e-13 #tests if the multiplication is correct
    qprod!(R_H,b2)
    @test norm((Q_H)*b - b2) <= 1e-13
    @test norm(F.Q - Q_H) <= 1e-13 #tests of unicity of QR decomposition
    @test norm(F.R - triu(R_H[1:n,1:n])) <= 1e-13
    @test norm(Q_H*triu(R_H) - A) <= 1e-13 #tests if the QR decomposition is correct
end

## Square matrices ##
@testset begin
    m, n = 10, 10
    A = rand(m,n)
    b = rand(m)
    b1 = deepcopy(b)
    b2 = deepcopy(b)
    R_H = deepcopy(A)
    qrH!(R_H)
    F = qr(A)
    
    Q_H = QRebuild!(R_H)

    @test norm(Q_H'Q_H - I) <= 1e-13 #tests that Q_H si unitary
    qtprod!(R_H,b1)
    @test norm((Q_H')*b - b1) <= 1e-13 #tests if the multiplication is correct
    @test norm((Q_H)*b2 - qprod!(R_H,b2)) <= 1e-13
    @test norm(F.Q - Q_H) <= 1e-13 #tests of unicity of QR decomposition
    @test norm(F.R - triu(R_H[1:n,1:n])) <= 1e-13
    @test norm(Q_H*triu(R_H) - A) <= 1e-13 #tests if the QR decomposition is correct
end