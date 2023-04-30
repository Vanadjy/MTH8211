include("HouseholderCompact.jl")
include("QR_ConcatHorizontal.jl")
include("blocdiagQR.jl")
include("MatricesTestSet.jl")
include("StructuredQR.jl")

using BundleAdjustmentModels, NLPModels

df = problems_df()
filter_df = df[ ( df.nequ .≥ 50000 ) .& ( df.nvar .≤ 34000 ), :]

name1 = filter_df[1, :name]
name2 = filter_df[2, :name]

model1 = BundleAdjustmentModel(name1)
model2 = BundleAdjustmentModel(name2)

A1 = jac_residual(model1, model1.meta.x0)
A2 = jac_residual(model2, model2.meta.x0)

#display(A1)
#display(A2)

qr(A1)

Householder_Compact!(A1)