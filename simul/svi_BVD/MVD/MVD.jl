mutable struct MVD
    K1::Int64
    K2::Int64
    Corpus1::Corpus
    Corpus2::Corpus
    Alpha::Matrix{Float64}
    B1::Matrix{Float64}
    B2::Matrix{Float64}
    Elog_B1::Matrix{Float64}
    Elog_B2::Matrix{Float64}
    Elog_Theta::MatrixList{Float64}
    γ::MatrixList{Float64}
    b1::Matrix{Float64}
    b2::Matrix{Float64}

#         model = MVD(K1, K2, Corpus1, Corpus2, init_params_...)
# MVD(K1, K2, Corpus1, Corpus2, Alpha,B1,B2,Elog_B1,Elog_B2,Elog_Theta,γ,b1,b2)
# Docs1 = [Document(Corpus1.Data[i], Corpus1.Data[i], length(Corpus1.Data[i])) for i in 1:Corpus1.N]
# Docs2 = [Document(Corpus2.Data[i], Corpus2.Data[i], length(Corpus2.Data[i])) for i in 1:Corpus2.N]
# c1 = Corpus(Docs1,Corpus1.N,Corpus1.V)
# c2 = Corpus(Docs2,Corpus2.N,Corpus2.V)
# Corpus1 = deepcopy(c1)
# Corpus2 = deepcopy(c2)
# end
end
