include("HouseholderCompact.jl")
include("QR_ConcatHorizontal.jl")
include("blocdiagQR.jl")

A_vect = CreateDiagBlock(4, 10, 10)
#F = blocdiagQR(A_vect)
A1 = CreateDiagBlockFull(4, 10, 10)[1]

m, n = size(A1)

A2 = rand(m, rand(1:(m-n)))

#L'idée maintenant c'est de faire du blocdiagQR sur A1, du QRHouseholder sur le bloc A2 et de faire du
#QR_concat_h sur ces deux blocs que l'on va concaténer horizontalement