m = 100
n1 = 50
n2 = 40
n = n1 + n2

A = rand(m, n)
b = rand(m)
A1, A2 = splitH(A, n1)

QRhcat!_a = @allocated begin
    QRhcat!(A1, A2)
end; @show(QRhcat!_a)