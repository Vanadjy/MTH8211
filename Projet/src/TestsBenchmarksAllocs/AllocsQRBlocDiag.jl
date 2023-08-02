nb_Matrix = 50
m_max = 50
n_max = 30

A_vect, m, n = CreateDiagBlock(nb_Matrix, m_max, n_max)

QRblocdiag!_a = @allocated begin
    QRblocdiag!(A_vect)
end; @show(QRblocdiag!_a)