# using BenchmarkTools
include("utils.jl")
include("init.jl")


function epoch_batches(N::Int64, mb_size::Int64, h_map::Vector{Bool})
	# mb_size=64
	N_ = N - sum(h_map)
	div_ = div(N_, mb_size)
	nb = (div_ * mb_size - N_) < 0 ? div_ + 1 : div_
	y = shuffle(collect(1:N)[.!h_map])
	x = [Int64[] for _ in 1:nb]
	for n in 1:nb
		while length(x[n]) < mb_size && !isempty(y)
			push!(x[n],pop!(y))
		end
	end
	return x, nb
end


function estimate_thetas(gamma)
	theta_est = deepcopy(gamma)
	# theta_est = deepcopy(γ)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
		# s = sum(γ[i])
		theta_est[i,:] ./= s
	end
	return theta_est
end
function estimate_B(b_)
	res = zeros(Float64, size(b_))
	for k in 1:size(b_, 1)
		res[k,:] .= mean(Dirichlet(b_[k,:]))
	end
	return res
end
function update_Elogtheta!(γ_, Elog_)
	for i in 1:size(Elog_,1)
		digsum = digamma_(sum(γ_[i]))
		Elog_[i,:] .=  digamma_.(γ_[i]) .- digsum
	end
end
function update_Elogtheta!(γ_, Elog_, mb_)
	for i in mb_
		digsum = digamma_(sum(γ_[i]))
		Elog_[i,:] .=  digamma_.(γ_[i]) .- digsum
	end
end

function update_Elogtheta_i(γ_, Elog_)
	digsum = digamma_(sum(γ_))
	Elog_[:] .=  digamma_.(γ_) .- digsum
	return Elog_[:]
end
function update_Elogb!(b_, Elog_)
	for k in 1:size(Elog_,1)
		digsum = digamma_(sum(b_[k,:]))
		Elog_[k,:] .=  digamma_.(b_[k,:]) .- digsum
	end
end

function optimize_γi!(K_, Alpha_,γ_, sum_phi_)
	for k in 1:K_
		γ_[k] = Alpha_[k] + sum_phi_[k]
	end
end

function optimize_b(len_mb, beta_prior, sum_phi_mb, V::Int64, K::Int64, N::Int64)
	b_ = ones(Float64, (K, V)) .* beta_prior
	b_ .+= (N/len_mb) .* sum_phi_mb
	return b_
end



function optimize_phi_iw(Elog_Theta_,Elog_B_, params_::CountParams, v)
	#####
	S = zeros(Float64, params_.K)
	S .+= Elog_Theta_[:]
	for k in 1:params_.K
		S[k] += Elog_B_[k,v]   #add vector to each row
	end
	S = deepcopy(softmax(S))
	return S
end


function get_lr(epoch, S, κ)
	###Should this be epoch based or iter based?
	return (S+epoch)^(-κ)
end


##### w , and v needs to be fixed in holdout ho and obs what to save probably their indices that correposnd
function calc_theta_bar_i(obs, Corpus,  i, γ, Alpha, Elog_Theta, Elog_B, count_params, phi,  w_in_phi)

	Elog_Theta[i,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:])
	doc = deepcopy(obs[i])
	corp = deepcopy(Corpus.Data[i])

	sum_phi_i = zeros(Float64, count_params.K)
	#############
	for _u in 1:10
		sum_phi_i = zeros(Float64, count_params.K)
		for val in unique(corp[w_in_phi[i]])
			x = findall(x -> x == val, corp[w_in_phi[i]])

			y = optimize_phi_iw(Elog_Theta[i,:],Elog_B, count_params,val)
			for xx in x
				phi[i][xx,:] .= y
			end
			sum_phi_i .+= length(x).*y
		end

		optimize_γi!(count_params.K, Alpha,γ[i], sum_phi_i)
		Elog_Theta[i,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:])
	end

	theta_bar = γ[i][:] ./ sum(γ[i])
	return theta_bar
end


function calc_perp(obs,ho,corpus, γ, Alpha, Elog_Theta,
 Elog_B, count_params, phi, w_in_phi,w_in_ho, B_est)
	corp = deepcopy(corpus)
	l = 0.0
	for i in collect(keys(ho))

		theta_bar = calc_theta_bar_i(obs,corp, i, γ, Alpha, Elog_Theta,
		 Elog_B, count_params, phi, w_in_phi)


		for w in w_in_ho[i]
			v = corp.Data[i][w]
			tmp = 0.0
			for k in 1:count_params.K
				tmp += ((B_est[k,v]*theta_bar[k]))
			end
			l += log(tmp)
		end

	end
	l/= sum(length.(collect(values(ho))))

	return exp(-l)
end
