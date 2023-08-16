export HcatMatrix, assemble

mutable struct HcatMatrix{A1<:Matrix, A2<:Matrix}
    leftm::A1
    rightm::A2
end

function assemble(A::HcatMatrix)
    # CAREFUL : when A is assembled, it is no longer of the type HcatMatrix !!
    return hcat(A.leftm, A.rightm)
end