include("utils.jl")
Random.seed!(1234)

function create_Alpha(K1, K2)
	"""
	Creates 	the Dirichlet prior
	Used in 	DGP and testing
	Returns  	the full vector and the matrix
	"""
    res = Float64[]
    while length(res) < K1*K2
        r = rand(Normal(1.0/(K1*K2)^.8, 1.0/(K1*K2)^1.6))
        if r > 0
            res = vcat(res, r)
        else
			continue
		end
    end
	res = normalize(res, 1)
    Res = convert(Matrix{Float64}, transpose(reshape(res, (K2, K1))))
    return res, Res
end


function create_Theta(vec, N, K1, K2)
	"""
	Creates 	Thetas for all individuals
				Given the Dirichlet prior vec
	Used in 	DGP and testing
	Returns 	vector Theta, Theta Matrix and Thetax
	"""
    res = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(vec),N)))
    Res = [convert(Matrix{Float64},transpose(reshape(res[i,:], (K2, K1)))) for i in 1:N]
    Res1 = [sum(Res[i], dims=2)[:,1] for i in 1:N]
    Res2 = [sum(Res[i], dims=1)[1,:] for i in 1:N]
    return res, Res, Res1, Res2
end

function create_B(beta, V, K)
	"""
	Creates 	B for all topics
				Given the Dirichlet prior beta
	Used in 	DGP
	Returns 	K*V matrix of vocab dist B
	"""
	B = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(repeat([beta], V)),K)))
    return B
end

function create_doc(wlen, topic_dist_vec ,term_topic_dist, Vocab, mode_, K1, K2)
	"""
	Creates 	document of a speicif wlen
				and topic and term topic dists
	Used in 	DGP
	Returns 	A doc vector of its words
	"""
	doc = String[]
	for i in 1:wlen
		topic_temp = rand(Distributions.Categorical(topic_dist_vec))
		row = Int64(ceil(topic_temp/K2))
		col = topic_temp - (row-1)*K2
		topic = mode_ == 1 ? row : col
		term = rand(Distributions.Categorical(term_topic_dist[topic,:]))
		doc = vcat(doc, Vocab[term])
	end
	return doc
end


function create_corpux(N, Theta_vec, B, K1, K2, wlens, Vocab, mode_)
	corpus = [String[] for i in 1:N]
	for i in 1:N
		doc  = create_doc(wlens[i], Theta_vec[i,:] ,B, Vocab, mode_, K1, K2)
		corpus[i] = vcat(corpus[i], doc)
	end
	return corpus
end

function prepare_data()
end

function init_param(N_, K1_, K2_, wlen1_, wlen2_,V1_, V2_, beta1_, beta2_)
	"""
	Initialize the relevant variables/params
	Set 	N, K1, K2, wlen1, wlen2
	Used in 	DGP and testing
	Returns	N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, γ, b1, b2,
			Vocab1 ,Vocab2, Beta1, Beta2,
			Corp1, Corp2
	"""
	N = N_
	K1 = K1_
	K2 = K2_
	alpha_vec, Alpha = create_Alpha(K1, K2)
	Theta_vec, Theta, Theta1, Theta2 = create_Theta(alpha_vec, N, K1, K2)

	fixed_len1 = wlen1_ .*ones(Int64, N)
	fixed_len2 = wlen2_ .*ones(Int64, N)

	phi1 = [zeros(Float64, (fixed_len1[i],K1, K2)) for i in 1:N]
	phi2 = [zeros(Float64, (fixed_len2[i],K1, K2)) for i in 1:N]

	γ = [zeros(Float64, K1, K2) for i in 1:N]
	for i in 1:N
		γ[i] = deepcopy(Alpha)
	end
	Vocab1 = ["term$x" for x in 1:V1_]
	Vocab2 = ["term$x" for x in 1:V2_]

	Beta1 = create_B(beta1_,V1_, K1_)
	Beta2 = create_B(beta2_,V2_, K2_)
	Corp1 = create_corpux(N, Theta_vec, Beta1, K1, K2, fixed_len1, Vocab1, 1)
	Corp2 = create_corpux(N, Theta_vec, Beta2, K1, K2, fixed_len2, Vocab2, 2)
	b1 = ones(Float64, (K1_, V1_)) .* beta1_
	b2 = ones(Float64, (K2_, V2_)) .* beta2_
	return N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, γ, b1, b2,
			Vocab1 ,Vocab2, Beta1, Beta2,
			Corp1, Corp2
end

function compute_ℒ_b1_full(K1, V1, beta1, b1)
	Beta1  = beta1 .* ones(Float64, (K1, V1))
	ℒ = 0.0
	for k in 1:K1
		ℒ += SpecialFunctions.lgamma(sum(Beta1[k,:]))
		ℒ -= SpecialFunctions.lgamma(sum(b1[k,:]))
		for v in 1:V1
			ℒ -= SpecialFunctions.lgamma(Beta1[k,v])
			ℒ += SpecialFunctions.lgamma(b1[k,v])
			ℒ += (Beta1[k,v]-b1[k,v]) * (digamma_(b1[k,v]) - digamma_(sum(b1[k,:])))
		end
	end
	return ℒ
end

function compute_ℒ_b2_full(K2, V2, beta2, b2)
	Beta2  = beta2 .* ones(Float64, (K2, V2))
	ℒ = 0.0
	for k in 1:K2
		ℒ += SpecialFunctions.lgamma(sum(Beta2[k,:]))
		ℒ -= SpecialFunctions.lgamma(sum(b2[k,:]))
		for v in 1:V2
			ℒ -= SpecialFunctions.lgamma(Beta2[k,v])
			ℒ += SpecialFunctions.lgamma(b2[k,v])
			ℒ += (Beta2[k,v]-b2[k,v]) * (digamma_(b2[k,v]) - digamma_(sum(b2[k,:])))
		end
	end
	return ℒ
end

function compute_ℒ_γ_full(N, Alpha, γ, K1, K2)
	ℒ = 0.0
	for i in 1:N
		ℒ += SpecialFunctions.lgamma(sum(Alpha))
		ℒ -= SpecialFunctions.lgamma(sum(γ[i]))
		for k1 in 1:K1
			for k2 in 1:K2
				ℒ -= SpecialFunctions.lgamma(Alpha[k1, k2])
				ℒ += SpecialFunctions.lgamma(γ[i][k1, k2])
				ℒ += (Alpha[k1, k2] - γ[i][k1, k2])*(digamma_(γ[i][k1,k2]) - digamma_(sum(γ[i])))
			end
		end
	end
	return ℒ
end

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

function compute_ℒ_phi2_full(N, Corp2, K1, K2, phi2, γ)
	ℒ = 0.0
	for i in 1:N
		for (w, val) in enumerate(Corp2[i])
			for k1 in 1:K1
				for k2 in 1:K2
					ℒ += phi2[i][w,k1, k2]*(digamma_(γ[i][k1,k2]) - digamma_(sum(γ[i])))
					ℒ -= phi2[i][w,k1, k2]*(log(phi2[i][w,k1, k2]))
				end
			end
		end
	end
	return ℒ
end

function compute_ℒ_y1_full(N, Corp1, K1, phi1, b1)
	ℒ = 0.0
	for i in 1:N
		for (w, val) in enumerate(Corp1[i])
			v = parse(Int64, val[5:end])
			for k1 in 1:K1
				ℒ += sum(phi1[i][w, k1, :])*(digamma_(b1[k1,v]) - digamma_(sum(b1[k1,:])))
			end
		end
	end
	return ℒ
end

function compute_ℒ_y2_full(N, Corp2, K2, phi2, b2)
	ℒ = 0.0
	for i in 1:N
		for (w, val) in enumerate(Corp2[i])
			v = parse(Int64, val[5:end])
			for k2 in 1:K2
				ℒ += sum(phi2[i][w, :, k2])*(digamma_(b2[k2,v]) - digamma_(sum(b2[k2,:])))
			end
		end
	end
	return ℒ
end

function compute_ℒ_full(N,K1,K2,V1,V2,beta1,beta2,b1,b2,
						Alpha,γ,Corp1,Corp2,phi1,phi2)
	"""
	Computes 	ELBO for
	Returns 	the ELBO()
	"""
	ℒ = 0.0
	ℒ += compute_ℒ_b1_full(K1, V1, beta1, b1)
	ℒ += compute_ℒ_b2_full(K2, V2, beta2, b2)
	ℒ += compute_ℒ_γ_full(N, Alpha, γ, K1, K2)
	ℒ += compute_ℒ_phi1_full(N, Corp1, K1, K2, phi1, γ)
	ℒ += compute_ℒ_phi2_full(N, Corp2, K1, K2, phi2, γ)
	ℒ += compute_ℒ_y1_full(N, Corp1, K1, phi1, b1)
	ℒ += compute_ℒ_y2_full(N, Corp2, K2, phi2, b2)
    return ℒ
end

function compute_ℒ_γ_atom(K1_, K2_, k1_, k2_, Alpha_, γ_, phi1_, phi2_)##gamma  and phis are i-indexed
	"""
	Computes 	ELBO for a gamma cell
	Returns 	the ELBO(gamma_k'k)
	"""
    ℒ_γ = 0.0
    ℒ_γ += (Alpha_[k1_, k2_]-γ_[k1_, k2_])*(digamma_(γ_[k1_, k2_]))
    for l1 in 1:K1_
        for l2 in 1:K2_
            ℒ_γ -= (Alpha_[l1, l2]-γ_[l1, l2])*digamma_(sum(γ_))
        end
    end
    ℒ_γ -= SpecialFunctions.lgamma(sum(γ_))
    ℒ_γ += (SpecialFunctions.lgamma(γ_[k1_, k2_]))

    for w in 1:size(phi1_,1)
        ℒ_γ += phi1_[w,k1_, k2_] * (digamma_(γ_[k1_,k2_]))
	end
    ℒ_γ -=  size(phi1_,1) * digamma_(sum(γ_))

	for w in 1:size(phi2_,1)
        ℒ_γ += phi2_[w,k1_, k2_] * (digamma_(γ_[k1_,k2_]))
	end
    ℒ_γ -=  size(phi2_,1) * digamma_(sum(γ_))
    return ℒ_γ
end
function compute_∇ℒ_γ_atom(K1_, K2_, k1_, k2_,Alpha_,γ_,phi1_, phi2_) # indexed at i
	"""
	Computes 	the gradient of ELBo w.r.t a gamma cell
	Returns 	the gradient and its parts
	"""
    rest = 0.0
    rest += (Alpha_[k1_, k2_]-γ_[k1_, k2_])*(trigamma_(γ_[k1_, k2_]))
    for w in 1:size(phi1_,1)
        rest += phi1_[w, k1_, k2_]*trigamma_(γ_[k1_, k2_])
    end
    for w in 1:size(phi2_,1)
        rest += phi2_[w, k1_, k2_]*trigamma_(γ_[k1_, k2_])
    end

    special_term = 0.0
    for l1 in 1:K1_
        for l2 in 1:K2_
            special_term += (Alpha_[l1, l2]-γ_[l1, l2])
        end
    end
    special_term += (size(phi1_,1)+size(phi2_,1))*1.0

	special_term *= -trigamma_(sum(γ_))
    ∇_γ = rest + special_term
    return ∇_γ , rest , special_term
end
function optimize_γ_ind(K1_, K2_, Alpha_,phi1_, phi2_)
	"""
	Optimize 	individual i's gamma for all its cells
	Returns 	optimized update for gamma
	"""
	γ_running_new =zeros(Float64, (K1_, K2_))
	for k1 in 1:K1_
		for k2 in 1:K2_
			sum1 = sum([phi1_[w, k1, k2] for w in 1:size(phi1_,1)])
			sum2 = sum([phi2_[w, k1, k2] for w in 1:size(phi2_,1)])
			γ_running_new[k1,k2] = Alpha_[k1, k2] + sum1 + sum2
		end
	end
	return γ_running_new
end

function update_γ(N_, K1_, K2_, γ_, Alpha_, phi1_, phi2_)
	"""
	updating all individual gammas
	updates		gamma in-place
	Returns		mean of the norm gradients
	"""
	norm_grad_results = zeros(Float64, N)
	for i in 1:N_
		# if i % 10 == 0
		# 	println(i)
		# end
		γ_[i] = optimize_γ_ind(K1_, K2_, Alpha_,phi1_[i], phi2_[i])

		norm_grad_results[i] = mean(norm.([compute_∇ℒ_γ_atom(K1_, K2_, k1, k2,Alpha_,γ_[i],phi1_[i], phi2_[i])[1]
		 for k1 in 1:K1_ for k2 in 1:K2_]))
	end
	return norm_grad_results
end
#### Each var param has these functions
#### 1)compute_ℒ_x_atom
#### 2)compute_∇ℒ_x_atom (not necessary if deterministic but for checks)
#### 3)optimize_x
#### 4)update_x (if deterministic can be coerced with 3)

function compute_ℒ_b_atom(beta, b, k, K, v, V_, doc, phi, dim)

	dig = digamma_(b[k,v])
	dig_sum = digamma_(sum(b[k,:]))
	lg = SpecialFunctions.lgamma(b[k,v])
	lg_sum = SpecialFunctions.lgamma_(sum(b[k,:]))

	ℒ_b = 0.0
	ℒ_b += (beta[k] - b[k,v]) * dig

	for v_ in 1:V_
		ℒ_b -= (beta[k] - b[k,v_]) * dig_sum
	end

	ℒ_b += (beta[k] - b[k,v_]) * lg

	for i in 1:N
		for (w, val) in enumerate(doc)
			V_val = parse(Int64, val[5:end])
			if dim == 1
				ℒ_b -= sum(phi[i][w,k, :])*dig_sum
			else
				ℒ_b -= sum(phi[i][w,:, k])*dig_sum
			end
			if V_val == v
				if dim == 1
					ℒ_b += sum(phi[i][w,k, :])*dig
				else
					ℒ_b += sum(phi[i][w,:, k])*dig
				end
			end
		end
	end
end

function optimize_b_vec!(N, b, beta, k, phi, dim, Corp, V)
	bk = beta .* ones(Float64, V)
	for i in 1:N
		doc = Corp[i]
		for (w, val) in enumerate(doc)
			v = parse(Int64, val[5:end])
			if dim == 1
				bk[v] += sum(phi[i][w,k, :])
			else
				bk[v] += sum(phi[i][w,:, k])
			end
		end
	end
	b[k,:] .= deepcopy(bk)
end


function optimize_phi_iw(phi_, γ_,b_, K1_, K2_, V, w, doc, dim)
	v = parse(Int64, doc[w][5:end])
	S = zeros(Float64, K1*K2)
	for k1 in 1:K1_
		for k2 in 1:K2_
			idx = (k1 - 1)*K2_ + k2
			S[idx] += (digamma_(γ_[k1, k2])) - (digamma_(sum(γ_)))
			if dim == 1
				S[idx] += (digamma_(b_[k1, v])) - (digamma_(sum(b_[k1,:])))
			else
				S[idx] += (digamma_(b_[k2, v])) - (digamma_(sum(b_[k2,:])))
			end
		end
	end
	S = deepcopy(softmax(S))
	phi_ = convert(Matrix{Float64}, transpose(reshape(S, (K2_, K1_))))
	return phi_
end

function estimate_thetas(gamma)
	theta_est = deepcopy(gamma)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
		theta_est[i] ./= s
	end
	return theta_est
end
function estimate_B(b)
	res = zeros(Float64, size(b))
	for k in 1:size(b, 1)
		s = sum(b[k,:])
		res ./= s
	end
	return res
end

# function updates()

	##init param
	beta1 = 0.08
	beta2 = 0.07
	wlen1 = 50
	wlen2 = 60
	N = 1000
	K1 = 3
	K2 = 5
	V1 = 200
	V2 = 300
	N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, γ, b1, b2,
			Vocab1 ,Vocab2, Beta1, Beta2,
			Corp1, Corp2 = init_param(N, K1, K2,wlen1, wlen2,V1, V2, beta1, beta2)

	MAX_ITER = 2000


	for iter in 1:MAX_ITER
		for i in 1:N
			doc1 = Corp1[i]
			doc2 = Corp2[i]
			for (w, val) in enumerate(doc1)
				phi1[i][w,:, :] = optimize_phi_iw(phi1[i], γ[i],b1, K1, K2, V1, w, doc1, 1)
			end
			for (w,val) in enumerate(doc2)
				phi2[i][w,:, :] = optimize_phi_iw(phi2[i], γ[i],b2, K1, K2, V2, w, doc2, 2)
			end
		end
		println(iter)
		update_γ(N, K1, K2, γ, Alpha, phi1, phi2)
		for k in 1:K1
			optimize_b_vec!(N, b1, beta1, k, phi1, 1, Corp1, V1)
		end
		for k in 1:K2
			optimize_b_vec!(N, b2, beta2, k, phi2, 2, Corp2, V2)
		end
		if (iter % 10 == 0) || (iter == 1)
			println(compute_ℒ_full(N,K1,K2,V1,V2,beta1,beta2,b1,b2,
								Alpha,γ,Corp1,Corp2,phi1,phi2))
		end
	end
	theta_est = estimate_thetas(γ)
	B1_est = estimate_B(b1)
	B2_est = estimate_B(b2)
	Plots.heatmap(theta_est[2])
	Plots.heatmap(Theta[2])
# end
sum(theta_est[6], dims=2)[:,1]
sum(theta_est[6], dims=1)[1,:]
Theta1[6]
Theta2[6]
