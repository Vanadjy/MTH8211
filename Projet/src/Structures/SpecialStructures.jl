export HcatMatrix

mutable struct HcatMatrix{A1<:AbstractMatrix{T}, A2<:AbstractMatrix{T}} where T
    A1::A1
    A2::A2
    function HcatMatrix(A1,A2)
        return hcat(A1,A2)
    end
end