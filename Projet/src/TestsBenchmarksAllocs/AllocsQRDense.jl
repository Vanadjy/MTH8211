m = 100
n1 = 50
n2 = 40
n = n1 + n2

A = rand(m, n)

qr!_a = @allocated begin
    LinearAlgebra.qr!(A)
end; @show(qr!_a)

qr!_a2 = @allocated begin
    GenericLinearAlgebra.qr!(A)
end; @show(qr!_a2)

qrH!_a = @allocated begin
    qrH!(A)
end; @show(qrH!_a)