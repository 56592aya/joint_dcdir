
# using BenchmarkTools
Random.seed!(1234)

"""
Creates 	the Dirichlet prior
\nUsed in 	DGP and testing
\nReturns  	the full vector and the matrix
"""
function create_Alpha(K1::Int64, K2::Int64)
	tot_dim = K1*K2
	mu_ = (inv(tot_dim))^.8; sd_ = (inv(tot_dim))^1.6;
	res = rand(Normal(mu_, sd_), tot_dim)
	Res = reshape(res, (K2, K1))
	Res = permutedims(Res, (2,1))
    return res, Res
end

"""
Creates 	Thetas for all individuals
	Given the Dirichlet prior vec
	\nUsed in 	DGP and testing
	\nReturns 	vector Theta, Theta Matrix
"""
function create_Theta(vec::Vector{Float64}, N::Int64, K1::Int64, K2::Int64)
	res = rand(Distributions.Dirichlet(vec),N)
	Res = [permutedims(reshape(res[:,i], (K2,K1)), (2,1)) for i in 1:N]
    return permutedims(res, (2, 1)), Res
end

"""
\nCreates 	B for all topics
Given the Dirichlet prior beta
\nUsed in 	DGP
\nReturns 	K*V matrix of vocab dist B
"""
function create_B(beta_prior::Matrix{Float64}, K::Int64, V::Int64)
	B = zeros(Float64, (K, V))
	for k in 1:K
		B[k,:] = rand(Distributions.Dirichlet(beta_prior[k,:]))
	end
    return B
end

"""
Creates 	document of a specific wlen
			and topic and term topic dists
\nUsed in 	DGP
\nReturns 	A doc vector of its words
"""
function create_doc(wlen::Int64, topic_dist_vec::Vector{Float64},
	                term_topic_dist::Matrix{Float64}, mode_::Int64,
					K1::Int64, K2::Int64)
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

"""
Creates 	corpus of a documents for a mode
\nUsed in 	DGP
\nReturns 	a list of lists
"""
function create_corpux(N::Int64, vec_list::Matrix{Float64}, B::Matrix{Float64},
	 				   K1::Int64, K2::Int64, wlens::Vector{Int64}, mode_::Int64)

	corpus = [Int64[] for i in 1:N]
	for i in 1:N
		doc  = create_doc(wlens[i], vec_list[i,:] ,B, mode_, K1, K2)
		corpus[i] = vcat(corpus[i], doc)
	end
	return corpus
end
"""
Creates		a simulated data
\nUsed in 	DGP
\nReturns 	Ground truth
\n=======
\nHow to choose data and parameters?
\nThings to decide on:
\nHow large you are choosing the vocabulary
\nHow large is each document
\nThere are some trade offs:
\nhow much overlap you have between the documents:
\nthis is the B prior
\nCoverage and overlap are in play
\nIf we want more definitve topics, then B prior should be small we either need
\n many many docs or somehow larger number of topics, specially with larger vocabulary
\nIf we want distrinct topics, increasing the number of documents alone is not going
\n to cut it, we need some topics as well.
\nWe don't want the V/wlens on average to be rediculously small
"""
function Create_Truth(N, K1, K2, V1, V2, β1_single, β2_single, wlen1_single, wlen2_single)
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


"""
Initialize  the relevant variables/params
\nUsed in 	testing
\nReturns		relevant parameters and containers
"""
function init_params(K1_::Int64, K2_::Int64, beta1_prior_, beta2_prior_,
	 alpha_prior_, corp1_, corp2_,V1_, V2_)
	#givens
	N = max(length(corp1_), length(corp2_))
	K1 = K1_
	K2 = K2_
	wlens1 = [length(corp1_[i]) for i in 1:N]
	wlens2 = [length(corp2_[i]) for i in 1:N]
	#priors
	alpha_vec = rand(Uniform(alpha_prior_/2,alpha_prior_*2), (K1*K2)) .* ones(Float64, K1*K2)
	Alpha =  permutedims(reshape(alpha_vec, (K2, K1)), (2,1))
	beta1 = ones(Float64, (K1,V1_)) .* rand(Uniform(beta1_prior_/4, beta1_prior_*2), K1)
	beta2 = ones(Float64, (K2,V2_)) .* rand(Uniform(beta2_prior_/4, beta2_prior_*2), K2)
	#variational params
	phi1 = [(1.0/(K1*K2)) .* ones(Float64, (wlens1[i], K1, K2)) for i in 1:N]
	phi2 = [(1.0/(K1*K2)) .* ones(Float64, (wlens2[i], K1, K2)) for i in 1:N]
	γ = [ones(Float64, (K1, K2)) for i in 1:N]
	# for i in 1:N
	# 	# γ[i] = deepcopy(Alpha)
	# 	# γ[i] = [ones(Float64, (K1, K2)) for i in 1:N]
	# end
	b1 = deepcopy(beta1)
	b2 = deepcopy(beta2)
	Elog_B1 = zeros(Float64, (K1, V1))
	Elog_B2 = zeros(Float64, (K2, V2))
	Elog_Theta = zeros(Float64, (N, K1, K2))
	## Also make sure vocab is the ones used.
	return N, K1, K2, wlens1, wlens2,
			alpha_vec, Alpha,beta1, beta2,
			phi1, phi2, γ, b1, b2,Elog_B1, Elog_B2,Elog_Theta
end
#####################   ESTIMATE  FUNNCS   ####################
"""
I use this because I want to feed a matrix
"""
function estimate_thetas(gamma)
	theta_est = deepcopy(gamma)
	# theta_est = deepcopy(γ)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
		# s = sum(γ[i])
		theta_est[i] ./= s
	end
	return theta_est
end
"""
Uses the Dirichlet mean
"""
function estimate_B(b_)
	res = zeros(Float64, size(b_))
	for k in 1:size(b_, 1)
		res[k,:] .= mean(Dirichlet(b_[k,:]))
	end
	return res
end
function update_Elogtheta!(γ_, Elog_)
	for i in 1:N
		digsum = digamma_(sum(γ_[i]))
		Elog_[i,:,:] .=  digamma_.(γ_[i]) .- digsum
	end
end
function update_Elogtheta_i(γ_, Elog_)
	digsum = digamma_(sum(γ_))
	Elog_[:,:] .=  digamma_.(γ_) .- digsum
	return Elog_[:,:]
end
function update_Elogb!(b_, Elog_)
	for k in 1:size(Elog_,1)
		digsum = digamma_(sum(b_[k,:]))
		Elog_[k,:] .=  digamma_.(b_[k,:]) .- digsum
	end
end

function gamma_converged(γ_, γ_old)
	val = mean(abs.(γ_ .- γ_old))
	if val < 1e-3
		println("mean change is $val")
		return true
	else
		return false
	end
end
###########  ELBO  ###############
"""
Returns the full elbo contribution of b
"""
function compute_ℒ_b_full(K_, V_, beta_prior_, b_)
	ℒ = 0.0
	for k in 1:K_
		ℒ -= SpecialFunctions.lgamma(sum(b_[k,:]))
		for v in 1:V_
			ℒ += SpecialFunctions.lgamma(b_[k,v])
			ℒ += (beta_prior_[k,v]-b_[k,v]) * (digamma_(b_[k,v]) - digamma_(sum(b_[k,:])))
		end
	end
	return ℒ
end
"""
Returns the full elbo contribution of gamma
"""
function compute_ℒ_γ_full(N, Alpha, γ, K1, K2)
	ℒ = 0.0
	for i in 1:N
		ℒ -= SpecialFunctions.lgamma(sum(γ[i]))
		for k1 in 1:K1
			for k2 in 1:K2
				ℒ += SpecialFunctions.lgamma(γ[i][k1, k2])
				ℒ += (Alpha[k1, k2] - γ[i][k1, k2])*(digamma_(γ[i][k1,k2]) - digamma_(sum(γ[i])))
			end
		end
	end
	return ℒ
end
"""
Returns the full elbo contribution of phi
"""
function compute_ℒ_phi1_full(N, Corp1, K1, K2, phi1, γ)
	ℒ = 0.0
	for i in 1:N
		for (w, val) in enumerate(Corp1[i])
			for k1 in 1:K1
				for k2 in 1:K2
					ℒ += phi1[i][w,k1, k2]*(digamma_(γ[i][k1,k2]) - digamma_(sum(γ[i])))
					ℒ -= phi1[i][w,k1, k2]*(log(phi1[i][w,k1, k2]))
				end
			end
		end
	end
	return ℒ
end
"""
Returns the full elbo contribution of phi
"""
function compute_ℒ_phi_full(N, Corp, K1, K2, phi, γ)
	ℒ = 0.0
	for i in 1:N
		for (w, val) in enumerate(Corp[i])
			for k1 in 1:K1
				for k2 in 1:K2
					ℒ += phi[i][w,k1, k2]*(digamma_(γ[i][k1,k2]) - digamma_(sum(γ[i])))
					ℒ -= phi[i][w,k1, k2]*(log(phi[i][w,k1, k2]))
				end
			end
		end
	end
	return ℒ
end
"""
Returns the full elbo contribution of y1
"""
function compute_ℒ_y1_full(N, Corp1, K1, phi1, b1)
	ℒ = 0.0
	for i in 1:N
		for (w, val) in enumerate(Corp1[i])
			for k1 in 1:K1
				ℒ += sum(phi1[i][w, k1, :])*(digamma_(b1[k1,val]) - digamma_(sum(b1[k1,:])))
			end
		end
	end
	return ℒ
end
"""
Returns the full elbo contribution of y2
"""
function compute_ℒ_y2_full(N, Corp2, K2, phi2, b2)
	ℒ = 0.0
	for i in 1:N
		for (w, val) in enumerate(Corp2[i])
			for k2 in 1:K2
				ℒ += sum(phi2[i][w, :, k2])*(digamma_(b2[k2,val]) - digamma_(sum(b2[k2,:])))
			end
		end
	end
	return ℒ
end

"""
Supposed to return the full elbo as summ of others
could be possibly full of crap
"""
function compute_ℒ_full(N,K1,K2,V1,V2,beta1_prior,beta2_prior,b1,b2,
						Alpha,γ,Corp1,Corp2,phi1,phi2)
	ℒ = 0.0
	ℒ += compute_ℒ_b_full(K1, V1, beta1_prior, b1)
	ℒ += compute_ℒ_b_full(K2, V2, beta2_prior, b2)
	ℒ += compute_ℒ_γ_full(N, Alpha, γ, K1, K2)
	ℒ += compute_ℒ_phi_full(N, Corp1, K1, K2, phi1, γ)
	ℒ += compute_ℒ_phi_full(N, Corp2, K1, K2, phi2, γ)
	# compute_ℒ_phi_full(N, corp2, K1, K2, phi2, γ)
	ℒ += compute_ℒ_y1_full(N, Corp1, K1, phi1, b1)
	ℒ += compute_ℒ_y2_full(N, Corp2, K2, phi2, b2)

    return ℒ
end

###########   OPTIMIZATION/VAR    UPDATES   ###############
"""
Optimize all atoms of γ
"""
function optimize_γ!(N, K1_, K2_, Alpha_,γ_, phi1_, phi2_)
	for i in 1:N
		for k1 in 1:K1_
			for k2 in 1:K2_
				γ_[i][k1, k2] = Alpha_[k1 ,k2] +sum(phi1_[i][:,k1, k2])+ sum(phi2_[i][:,k1, k2])
			end
		end
	end
end
###
function optimize_γi!(K1_, K2_, Alpha_,γ_, phi1_, phi2_)
	for k1 in 1:K1_
		for k2 in 1:K2_
			γ_[k1, k2] = Alpha_[k1 ,k2] + sum(phi1_[:,k1, k2])+ sum(phi2_[:,k1, k2])
		end
	end
end
"""
Optimize all b1 per topic
"""
function optimize_b1_per_topic!(N, b, beta_prior, k, phi, corp, V)
	bk = deepcopy(beta_prior[k,:])
	for i in 1:N
		# doc = deepcopy(corp[i])
		for (w,val) in enumerate(corp[i])
			bk[corp[i][w]] += sum(phi[i][w,k, :])
		end
	end
	b[k,:] = bk
end

"""
Optimize all b2 per topic
"""
function optimize_b2_per_topic!(N, b, beta_prior, k, phi, corp, V)
	bk = deepcopy(beta_prior[k,:])
	for i in 1:N
		for (w,val) in enumerate(corp[i])
			bk[corp[i][w]] += sum(phi[i][w,:, k])
		end
	end
	b[k,:] = bk
end
"""
Optimize all b1
"""
function optimize_b1(N, beta_prior, phi, corp, K,V)
	b_ = ones(Float64, (K, V)) .* beta_prior
	for k in 1:K
		for i in 1:N
			doc = corp[i]
			for (w,val) in enumerate(doc)
				b_[k,val] += sum(phi[i][w,k, :])
			end
		end
	end
	return b_
end
"""
Optimize all b2
"""
function optimize_b2(N, beta_prior, phi, corp, K, V)
	b_ = ones(Float64, (K, V)) .* beta_prior
	for k in 1:K
		for i in 1:N
			doc = corp[i]
			for (w,val) in enumerate(doc)
				b_[k,val] += sum(phi[i][w,:,k])
			end
		end
	end
	return b_
end
"""
Optimize all phi atoms
"""
function optimize_phi1_iw(phi_, Elog_Theta_,Elog_B1_, K1_, K2_, w, doc)
	#####
	v = doc[w]
	S = zeros(Float64, (K1_,K2_))
	S .+= Elog_Theta_[:,:]
	for k in 1:K2_
		S[:,k] .+= Elog_B1_[:,v]   #add vector to each row
	end
	S = deepcopy(softmax(S))
	phi_ = S
	return phi_

end

function optimize_phi2_iw(phi_, Elog_Theta_,Elog_B2_, K1_, K2_, w, doc)
	#####
	v = doc[w]

	S = zeros(Float64, (K1_,K2_))
	S .+= Elog_Theta_[:,:]
	for k in 1:K1_
		S[k,:] .+= Elog_B2_[:,v]   #add vector to each row
	end
	S = deepcopy(softmax(S))
	phi_ = S
	return phi_

end



println("All funcs are loaded")
