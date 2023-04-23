using LinearAlgebra

include("blocdiagQR.jl")
include("QR_ConcatHorizontal.jl")

nb_mat = 1:10
M = 1:100

A_vect_collection = []
B_collection = []

for A in nb_mat
    for m in M
        for n = m:50
            A_vect, m_aux, n_aux = CreateDiagBlock(A, m, n)
            if m_aux ≥ n_aux && (m_aux - n_aux - 1) > 0 #assure d'avoir que des matrices surdéterminées
                B = rand(m_aux, m_aux - n_aux - 1)
                push!(A_vect_collection, A_vect)
                push!(B_collection, B)
            end
        end
    end
end
