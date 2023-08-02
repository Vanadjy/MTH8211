m = 100
n1 = 50
n2 = 40
n = n1 + n2

A = rand(m, n)
b = rand(m)
B = rand(m,n)
A1, A2 = splitH(A, n1)

R = copy(A)
qrH!(R)

println("Benchmark for qtprod! :")
display(@benchmark qtprod!($R,$b))
println("---------------")

println("Benchmark for qprod! :")
display(@benchmark qprod!($R,$b))
println("---------------")

println("Benchmark for qtmul! :")
display(@benchmark qtmul!($R,$B))
println("---------------")

println("Benchmark for qmul! :")
display(@benchmark qmul!($R,$B))
println("---------------")