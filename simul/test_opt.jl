include("utils.jl")
Random.seed!(1234)
function create_Alpha(K1, K2)
	"""
	Creates 	the Dirichlet prior
	Returns  	the full vector and the matrix
	"""
    res = Float64[]
    while length(res) < K1*K2
        # r = rand(Normal(1.0/(K1*K2), 1.0/(K1*K2)^2))
        r = rand(Normal(1.0/(K1*K2)^.8, 1.0/(K1*K2)^1.6))
		# r = 0.1 + rand(Normal(0.0, 0.01))
        # r = rand(1)[1]
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
	Returns 	vector Theta, Theta Matrix and Thetax
	"""
    res = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(vec),N)))
    Res = [convert(Matrix{Float64},transpose(reshape(res[i,:], (K2, K1)))) for i in 1:N]
    Res1 = [sum(Res[i], dims=2)[:,1] for i in 1:N]
    Res2 = [sum(Res[i], dims=1)[1,:] for i in 1:N]
    return res, Res, Res1, Res2
end

function create_phis!(phi, Thetax, N)
	"""
	Creates 	the variational phis for word topic distribution
	Returns 	nothing and creates phis in-place
	"""
    for i in 1:N
		wlen, K = size(phi[i])
        for w in 1:wlen
            s = sum(rand(Distributions.Multinomial(1,Thetax[i]), 20), dims=2)[:,1] .*1.0
            phi[i][w,:] = normalize(s,1)
        end
    end
end

function create_phis_2!(phi, Theta, N)
	"""
	Creates 	the variational phis for word topic distribution
	Returns 	nothing and creates phis in-place
	"""
    for i in 1:N

		wlen, K1, K2 = size(phi[i])
        for w in 1:wlen
            s = sum(rand(Distributions.Multinomial(1,Theta_vec[i,:]), 1000), dims=2)[:,1] .*1.0
			s = normalize(s,1)
			s = convert(Matrix{Float64}, transpose(reshape(s, (K2, K1))))
            phi[i][w,:,:] = s
        end
    end
end

function create_phis_3!(phi, Theta, N)
	"""
	Creates 	the variational phis for word topic distribution
	Returns 	nothing and creates phis in-place
	"""
    for i in 1:N
		wlen, K1, K2 = size(phi[i])
        for w in 1:wlen
            s = sum(rand(Distributions.Multinomial(1,Theta_vec[i,:]), 1000), dims=2)[:,1] .*1.0
			s = normalize(s,1)
			s = convert(Matrix{Float64}, transpose(reshape(s, (K2, K1))))
            phi[i][w,:,:] = s
        end
    end
end

function init_param(N_, K1_, K2_, wlen1_, wlen2_)
	"""
	Initialize the relevant variables/params
	Set 	N, K1, K2, wlen1, wlen2
	Returns	N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, gamma
	"""
	N = N_
	K1 = K1_
	K2 = K2_
	alpha_vec, Alpha = create_Alpha(K1, K2)
	Theta_vec, Theta, Theta1, Theta2 = create_Theta(alpha_vec, N, K1, K2)

	fixed_len1 = wlen1_ .*ones(Int64, N)
	fixed_len2 = wlen2_ .*ones(Int64, N)

	phi1 = [zeros(Float64, (fixed_len1[i],K1)) for i in 1:N]
	phi2 = [zeros(Float64, (fixed_len2[i],K2)) for i in 1:N]
	create_phis!(phi1, Theta1, N)
	create_phis!(phi2, Theta2, N)
	println("phis are created")

	γ = [zeros(Float64, K1, K2) for i in 1:N]
	for i in 1:N
		γ[i] = deepcopy(Alpha)
	end
	return N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, γ
end

function init_param_2(N_, K1_, K2_, wlen1_, wlen2_)
	"""
	Initialize the relevant variables/params
	Set 	N, K1, K2, wlen1, wlen2
	Returns	N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, gamma
	"""
	N = N_
	K1 = K1_
	K2 = K2_
	alpha_vec, Alpha = create_Alpha(K1, K2)
	Theta_vec, Theta, Theta1, Theta2 = create_Theta(alpha_vec, N, K1, K2)

	fixed_len1 = wlen1_ .*ones(Int64, N)
	fixed_len2 = wlen2_ .*ones(Int64, N)

	phi1 = [zeros(Float64, (fixed_len1[i],K1, K2)) for i in 1:N]
	create_phis_2!(phi1, Theta, N)
	phi2 = [zeros(Float64, (fixed_len2[i],K1, K2)) for i in 1:N]
	create_phis_2!(phi2, Theta, N)
	println("phis are created")

	γ = [zeros(Float64, K1, K2) for i in 1:N]
	for i in 1:N
		γ[i] = deepcopy(Alpha)
	end
	return N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, γ
end

function init_param_3(N_, K1_, K2_, wlen1_, wlen2_)
	"""
	Initialize the relevant variables/params
	Set 	N, K1, K2, wlen1, wlen2
	Returns	N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, gamma
	"""
	N = N_
	K1 = K1_
	K2 = K2_
	alpha_vec, Alpha = create_Alpha(K1, K2)
	Theta_vec, Theta, Theta1, Theta2 = create_Theta(alpha_vec, N, K1, K2)

	fixed_len1 = wlen1_ .*ones(Int64, N)
	fixed_len2 = wlen2_ .*ones(Int64, N)

	phi1 = [zeros(Float64, (fixed_len1[i],K1, K2)) for i in 1:N]
	create_phis_2!(phi1, Theta, N)
	phi2 = [zeros(Float64, (fixed_len2[i],K1, K2)) for i in 1:N]
	create_phis_2!(phi2, Theta, N)
	println("phis are created")

	γ = [zeros(Float64, K1, K2) for i in 1:N]
	for i in 1:N
		γ[i] = deepcopy(Alpha)
	end
	return N, K1, K2, alpha_vec, Alpha,
			Theta_vec, Theta, Theta1, Theta2,
			fixed_len1, fixed_len2,
			phi1, phi2, γ
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
        ℒ_γ += phi1_[w,k1_] * (digamma_(sum(γ_[k1_,:])))
        for l1 in 1:K1_
            ℒ_γ -= phi1_[w,l1] * digamma_(sum(γ_))
        end
    end
    for w in 1:size(phi2_,1)
        ℒ_γ += phi2_[w,k2_]*(digamma_(sum(γ_[:,k2_])))
        for l2 in 1:K2_
            ℒ_γ -= phi2_[w,l2] * digamma_(sum(γ_))
        end
    end
    return ℒ_γ
end
function compute_ℒ_γ_atom_2(K1_, K2_, k1_, k2_, Alpha_, γ_, phi1_, phi2_)##gamma  and phis are i-indexed
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


function compute_ℒ_γ_atom_3(K1_, K2_, k1_, k2_, Alpha_, γ_, phi1_, phi2_)##gamma  and phis are i-indexed
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
        rest += phi1_[w,k1_]*trigamma_(sum(γ_[k1_,:]))
    end
    for w in 1:size(phi2_,1)
        rest += phi2_[w,k2_]*trigamma_(sum(γ_[:,k2_]))
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

function compute_∇ℒ_γ_atom_2(K1_, K2_, k1_, k2_,Alpha_,γ_,phi1_, phi2_) # indexed at i
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

function compute_∇ℒ_γ_atom_3(K1_, K2_, k1_, k2_,Alpha_,γ_,phi1_, phi2_) # indexed at i
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


function optimize_γ_i(K1_, K2_,old_, Alpha_,phi1_, phi2_, ITER)
	"""
	Optimize 	individual i's gamma for all its cells
	Returns 	optimized update for gamma
	"""
	γ_running = deepcopy(old_)
	γ_running_new =deepcopy(γ_running)
	for iter in 1:ITER
		for k1 in 1:K1_
			sum1 = sum([phi1_[w,k1] for w in 1:size(phi1_,1)])
			for k2 in 1:K2_
				sum2 = sum([phi2_[w,k2] for w in 1:size(phi2_,1)])
				γ_running_new[k1,k2] = Alpha_[k1, k2] + sum1 * (γ_running[k1, k2]/sum(γ_running[k1,:]))+
					sum2 * (γ_running[k1, k2]/sum(γ_running[:,k2]))
				##Comment γ_running_new above and added below
				# s1 = sum([exp(digamma_(γ_running[k1_,k2])) for k1_ in 1:K1_])
				# s2 = sum([exp(digamma_(γ_running[k1,k2_])) for k2_ in 1:K2_])
				# val = exp(digamma_(γ_running[k1,k2]))
				# γ_running_new[k1,k2] = Alpha_[k1, k2] + (val/s2) * sum1 +
				# (val/s1)  * sum2
				##Added above
			end
		end
		γ_running=deepcopy(γ_running_new)
	end
	return γ_running

end


function optimize_γ_i_2(K1_, K2_,old_, Alpha_,phi1_, phi2_, ITER)
	"""
	Optimize 	individual i's gamma for all its cells
	Returns 	optimized update for gamma
	"""
	γ_running = deepcopy(old_)
	γ_running_new =deepcopy(γ_running)

	# phi1_ = phi1[1]
	# phi2_ = phi2[1]
	# K1_ = K1;K2_ = K2; old_ = deepcopy(Alpha); Alpha_ = deepcopy(Alpha)
	for iter in 1:ITER
		for k1 in 1:K1_
			for k2 in 1:K2_
				sum1 = sum([phi1_[w, k1, k2] for w in 1:size(phi1_,1)])
				sum2 = sum([phi2_[w, k1, k2] for w in 1:size(phi2_,1)])
				# γ_running_new[k1,k2] = Alpha_[k1, k2] + sum1 * (γ_running[k1, k2]/sum(γ_running[k1,:]))+
				# 	sum2 * (γ_running[k1, k2]/sum(γ_running[:,k2]))
				γ_running_new[k1,k2] = Alpha_[k1, k2] + sum1 + sum2
			end
		end
		γ_running=deepcopy(γ_running_new)
	end
	return γ_running
end

function optimize_γ_i_3(K1_, K2_, Alpha_,phi1_, phi2_)
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



function draw_prior_grad()
	"""
	Plots 	the gradient prior to optimization
	Returns nothing but a plot
	"""
	prior_grad_results = zeros(Float64, N)
	for i in 1:N
		prior_grad_results[i] = mean([compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])[1]
		 for k1 in 1:K1 for k2 in 1:K2])
	end
	Plots.histogram(prior_grad_results)
end

function draw_prior_grad_2()
	"""
	Plots 	the gradient prior to optimization
	Returns nothing but a plot
	"""
	prior_grad_results = zeros(Float64, N)
	for i in 1:N
		prior_grad_results[i] = mean([compute_∇ℒ_γ_atom_2(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])[1]
		 for k1 in 1:K1 for k2 in 1:K2])
	end
	Plots.histogram(prior_grad_results)
end

function draw_prior_grad_3()
	"""
	Plots 	the gradient prior to optimization
	Returns nothing but a plot
	"""
	prior_grad_results = zeros(Float64, N)
	for i in 1:N
		prior_grad_results[i] = mean([compute_∇ℒ_γ_atom_3(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])[1]
		 for k1 in 1:K1 for k2 in 1:K2])
	end
	Plots.histogram(prior_grad_results)
end

function update_γ(ITER, N_, K1_, K2_, γ_, Alpha_, phi1_, phi2_)
	"""
	updating all individual gammas
	updates		gamma in-place
	Returns		mean of the norm gradients
	"""
	norm_grad_results = zeros(Float64, N)
	for i in 1:N_
		if i % 10 == 0
			println(i)
		end
		γ_[i] = optimize_γ_i(K1_, K2_,Alpha_, Alpha_,phi1_[i], phi2_[i], ITER)

		norm_grad_results[i] = mean(norm.([compute_∇ℒ_γ_atom(K1_, K2_, k1, k2,Alpha_,γ_[i],phi1_[i], phi2_[i])[1]
		 for k1 in 1:K1_ for k2 in 1:K2_]))
	end
	return norm_grad_results
end
function update_γ_2(ITER, N_, K1_, K2_, γ_, Alpha_, phi1_, phi2_)
	"""
	updating all individual gammas
	updates		gamma in-place
	Returns		mean of the norm gradients
	"""
	norm_grad_results = zeros(Float64, N)
	for i in 1:N_
		if i % 10 == 0
			println(i)
		end
		γ_[i] = optimize_γ_i_2(K1_, K2_,Alpha_, Alpha_,phi1_[i], phi2_[i], ITER)

		norm_grad_results[i] = mean(norm.([compute_∇ℒ_γ_atom_2(K1_, K2_, k1, k2,Alpha_,γ_[i],phi1_[i], phi2_[i])[1]
		 for k1 in 1:K1_ for k2 in 1:K2_]))
	end
	return norm_grad_results
end

function update_γ_3(N_, K1_, K2_, γ_, Alpha_, phi1_, phi2_)
	"""
	updating all individual gammas
	updates		gamma in-place
	Returns		mean of the norm gradients
	"""
	norm_grad_results = zeros(Float64, N)
	for i in 1:N_
		if i % 10 == 0
			println(i)
		end
		γ_[i] = optimize_γ_i_3(K1_, K2_, Alpha_,phi1_[i], phi2_[i])

		norm_grad_results[i] = mean(norm.([compute_∇ℒ_γ_atom_3(K1_, K2_, k1, k2,Alpha_,γ_[i],phi1_[i], phi2_[i])[1]
		 for k1 in 1:K1_ for k2 in 1:K2_]))
	end
	return norm_grad_results
end
