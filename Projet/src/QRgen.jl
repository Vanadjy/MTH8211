using BlockDiagonals, LinearAlgebra, SparseArrays

qrgen!(A::AbstractMatrix) = qrH!(A) #Dense matrix

qrgen!(A::AbstractSparseMatrix) = LinearAlgebra.qr!(A) #Sparse matrix

qrgen!(A::BlockDiagonal{T, Matrix{T}}) where T = QRblocdiag!(blocks(A)) #BlockDiagonal matrix

#qrgen!(A::HcatMatrix{AbstractMatrix{T}, AbstractMatrix{T}}) where T = QRhcat!(A.A1, A.A2) #Horizontally concatenated matrix