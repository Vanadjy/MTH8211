## Tests if hte function QRhat uses the correct QR factorisation according to the type of the matrix ##
println("-------------------------")
@testset begin
    m, n = 10, 8
    A = rand(m,n)
    R_H = deepcopy(A)
    qrhat!(R_H)
    
    nb_Matrix = 50
    m_max = 50
    n_max = 30

    A_vect, m, n = CreateDiagBlock(nb_Matrix, m_max, n_max)
    R_vect = deepcopy(A_vect)
    bm = BlockDiagonal(R_vect)
    qrhat!(bm)

    H = HcatMatrix(rand(10,4), rand(10,5))
    println(typeof(H))
    qrhat!(H)
end