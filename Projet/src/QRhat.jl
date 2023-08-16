export qrhat!, qrhat

qrhat!(A::AbstractMatrix) = qrH!(A) #Dense matrix

qrhat!(A::AbstractSparseMatrix) = LinearAlgebra.qr!(A) #Sparse matrix

qrhat!(A::BlockDiagonal) = QRblocdiag!(A) #BlockDiagonal matrix

qrhat!(A::AbstractBlockMatrix{T}) where T = QRhcat!(A) #Horizontally concatenated matrix

############

function qrhat(A::Union{AbstractMatrix, AbstractSparseMatrix, BlockDiagonal{T, Matrix{T}}, AbstractBlockMatrix{T}}) where T
    B = copy(A)
    qrhat!(B)
    B
end 