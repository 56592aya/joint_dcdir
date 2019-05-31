include("utils.jl")
using BenchmarkTools
Random.seed!(1234)

function create_Alpha(K1::Int64, K2::Int64)
	"""
	Creates 	the Dirichlet prior
	Used in 	DGP and testing
	Returns  	the full vector and the matrix
	"""
	tot_dim = K1*K2
	mu_ = (inv(tot_dim))^.8; sd_ = (inv(tot_dim))^1.6;
	res = rand(Normal(mu_, sd_), tot_dim)
	Res = reshape(res, (K2, K1))
	Res = permutedims(Res, (2,1))
    return res, Res
end

function create_Theta(vec::Vector{Float64}, N::Int64, K1::Int64, K2::Int64)
	"""
	Creates 	Thetas for all individuals
				Given the Dirichlet prior vec
	Used in 	DGP and testing
	Returns 	vector Theta, Theta Matrix
	"""
	res = rand(Distributions.Dirichlet(vec),N)
	Res = [permutedims(reshape(res[:,i], (K2,K1)), (2,1)) for i in 1:N]
    return permutedims(res, (2, 1)), Res
end

function create_B(beta_prior::Matrix{Float64}, K::Int64, V::Int64)
	"""
	Creates 	B for all topics
				Given the Dirichlet prior beta
	Used in 	DGP
	Returns 	K*V matrix of vocab dist B
	"""
	B = zeros(Float64, (K, V))
	for k in 1:K
		B[k,:] = rand(Distributions.Dirichlet(beta_prior[k,:]))
	end
    return B
end

function create_doc(wlen::Int64, topic_dist_vec::Vector{Float64},
	                term_topic_dist::Matrix{Float64}, mode_::Int64,
					K1::Int64, K2::Int64)
	"""
	Creates 	document of a specific wlen
				and topic and term topic dists
	Used in 	DGP
	Returns 	A doc vector of its words
	"""
	doc = Int64[]
	for w in 1:wlen
		topic_temp = rand(Distributions.Categorical(topic_dist_vec))
		row = Int64(ceil(topic_temp/K2))
		col = topic_temp - (row-1)*K2
		topic = mode_ == 1 ? row : col
		term = rand(Distributions.Categorical(term_topic_dist[topic,:]))
		doc = vcat(doc, term)
	end
	return doc
end


function create_corpux(N::Int64, vec_list::Matrix{Float64}, B::Matrix{Float64},
	 				   K1::Int64, K2::Int64, wlens::Vector{Int64}, mode_::Int64)
	"""
	Creates 	corpus of a documents for a mode
	Used in 	DGP
	Returns 	a list of lists
	"""
	corpus = [Int64[] for i in 1:N]
	for i in 1:N
		doc  = create_doc(wlens[i], vec_list[i,:] ,B, mode_, K1, K2)
		corpus[i] = vcat(corpus[i], doc)
	end
	return corpus
end

function Create_Truth(N, K1, K2, V1, V2, β1_single, β2_single, wlen1_single, wlen2_single)
	"""
	Creates		a simulated data
	Used in 	DGP
	Returns 	Ground truth
	"""
	α, Α = create_Alpha(K1, K2)
	θ,Θ = create_Theta(α, N, K1, K2)
	β1 = ones(Float64, (K1, V1)) .* β1_single
	Β1 = create_B(β1, K1, V1)
	β2 = ones(Float64, (K2, V2)) .* β2_single
	Β2 = create_B(β2, K2, V2)
	wlens1 = [wlen1_single for i in 1:N]
	wlens2 = [wlen2_single for i in 1:N]
	corp1 = create_corpux(N, θ, Β1,K1,K2, wlens1, 1)
	corp2 = create_corpux(N, θ, Β2,K1,K2, wlens2, 2)
	return α,Α, θ,Θ, Β1, Β2, β1, β2, V1, V2, corp1, corp2
end


#####################
function init_params(K1_, K2_, beta1_prior_, beta2_prior_, alpha_prior_, corp1_, corp2_)
	"""
	Initialize  the relevant variables/params
	Used in 	testing
	Returns		relevant parameters and containers
	"""
	N = maximum(length(corp1_), length(corp2_))
	K1 = K1_
	K2 = K2_
	alpha_vec, Alpha = create_Alpha(K1, K2)
	wlens1 = [length(corp1_[i]) for i in 1:N]
	wlens2 = [length(corp2_[i]) for i in 1:N]


	phi1 = [(1.0/(K1*K2)) .* ones(Float64, (wlens1[i], K1, K2)) for i in 1:N]
	phi2 = [(1.0/(K1*K2)) .* ones(Float64, (wlens2[i], K1, K2)) for i in 1:N]

	γ = [zeros(Float64, (K1, K2)) for i in 1:N]
	for i in 1:N
		γ[i] = deepcopy(Alpha)
	end
	Vocab1 = collect(1:V1_)
	Vocab2 = collect(1:V2_)

	beta1 =
	beta2 =
	b1 = deepcopy(beta1_prior_)
	b2 = deepcopy(beta2_prior_)
	return N, K1, K2, alpha_vec, Alpha,
			wlens1, wlens2,
			phi1, phi2, γ, b1, b2,
			Vocab1 ,Vocab2, beta1, beta2
end
