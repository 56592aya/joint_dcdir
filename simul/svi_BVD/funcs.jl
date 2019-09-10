# using BenchmarkTools
include("dgp.jl")
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

##to remove:
# mb_size = 100
# @btime epoch_batches($N, $mb_size)

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
		Elog_[i,:,:] .=  digamma_.(γ_[i]) .- digsum
	end
end
function update_Elogtheta!(γ_, Elog_, mb_)
	for i in mb_
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

function optimize_γ!(mb_, mindex_, K1_, K2_, Alpha_,γ_, phi1_, phi2_)
	for i in mb_[mindex_]
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
function optimize_γi_2!(K1_, K2_, Alpha_,γ_, sum_phi1_, sum_phi2_)
	for k1 in 1:K1_
		for k2 in 1:K2_
			γ_[k1, k2] = Alpha_[k1 ,k2] + sum_phi1_[k1, k2] + sum_phi2_[k1, k2]
		end
	end
end
"""
Optimize all b1 per topic
"""
function optimize_b1_per_topic!(mb_, b, beta_prior, k, phi, params_::CountParams, corpus_::Corpus)
	bk = deepcopy(beta_prior[k,:])
	for i in mb_
		doc = deepcopy(corpus_.Data[i])
		for (w,val) in enumerate(doc)
			bk[doc[w]] += (params_.N/length(mb_[mindex_]))*sum(phi[i][w,k, :])
		end
	end
	b[k,:] = bk
end
"""
Optimize all b2 per topic
"""
function optimize_b2_per_topic!(mb_, b, beta_prior, k, phi, params_::CountParams, corpus_::Corpus)
	bk = deepcopy(beta_prior[k,:])
	for i in mb_
		doc = deepcopy(corpus_.Data[i])
		for (w,val) in enumerate(doc)
			bk[doc[w]] += (params_.N/length(mb_[mindex_]))*sum(phi[i][w,:, k])
		end
	end
	b[k,:] = bk
end
"""
Optimize all b1
"""
function optimize_b1(mb_, beta_prior, phi, corpus_::Corpus, params_::CountParams)
	b_ = ones(Float64, (params_.K1, corpus_.V)) .* beta_prior
	for k in 1:params_.K1
		for i in mb_
			doc = corpus_.Data[i]
			for (w,val) in enumerate(doc)
				b_[k,val] += (params_.N/length(mb_))*sum(phi[i][w,k, :])
			end
		end
	end
	return b_
end
function optimize_b(len_mb, beta_prior, sum_phi_mb, V::Int64, K::Int64, N::Int64)
	b_ = ones(Float64, (K, V)) .* beta_prior
	b_ .+= (N/len_mb) .* sum_phi_mb
	return b_
end
"""
Optimize all b2
"""
function optimize_b2(mb_, beta_prior, phi, corpus_::Corpus, params_::CountParams)
	b_ = ones(Float64, (params_.K1, corpus_.V)) .* beta_prior
	for k in 1:params_.K2
		for i in mb_
			doc = corpus_.Data[i]
			for (w,val) in enumerate(doc)
				b_[k,val] += (params_.N/length(mb_))*sum(phi[i][w,:,k])
			end
		end
	end
	return b_
end
# function optimize_b1_2(len_mb, beta_prior, sum_phi_mb, V::Int64, K::Int64, N::Int64)
# 	b_ = ones(Float64, (K, V)) .* beta_prior
# 	for k in 1:K
# 		for v in 1:V
# 			b_[k,v] += (N/len_mb)*sum_phi_mb[k,v]
# 		end
# 	end
# 	return b_
# end
"""
Optimize all phi atoms
"""
function optimize_phi1_iw(phi_, Elog_Theta_,Elog_B1_, params_::CountParams, w, doc)
	#####
	v = doc[w]
	S = zeros(Float64, (params_.K1,params_.K2))
	S .+= Elog_Theta_[:,:]
	for k in 1:params_.K2
		S[:,k] .+= Elog_B1_[:,v]   #add vector to each row
	end
	S = deepcopy(softmax(S))
	phi_ = S
	return phi_
end
function optimize_phi1_iw_2(Elog_Theta_,Elog_B1_, params_::CountParams, v)
	#####
	S = zeros(Float64, (params_.K1,params_.K2))
	S .+= Elog_Theta_[:,:]
	for k in 1:params_.K2
		S[:,k] .+= Elog_B1_[:,v]   #add vector to each row
	end
	S = deepcopy(softmax(S))
	return S
end

function optimize_phi2_iw(phi_, Elog_Theta_,Elog_B2_, params_::CountParams, w, doc)
	#####
	v = doc[w]

	S = zeros(Float64, (params_.K1,params_.K2))
	S .+= Elog_Theta_[:,:]
	for k in 1:params_.K1
		S[k,:] .+= Elog_B2_[:,v]   #add vector to each row
	end
	S = deepcopy(softmax(S))
	phi_ = S
	return phi_
end
function optimize_phi2_iw_2(Elog_Theta_,Elog_B2_, params_::CountParams,v)
	#####

	S = zeros(Float64, (params_.K1,params_.K2))
	S .+= Elog_Theta_[:,:]
	for k in 1:params_.K1
		S[k,:] .+= Elog_B2_[:,v]   #add vector to each row
	end
	S = deepcopy(softmax(S))
	return S
end

function get_lr(epoch, S, κ)
	###Should this be epoch based or iter based?
	return (S+epoch)^(-κ)
end


##### w , and v needs to be fixed in holdout ho and obs what to save probably their indices that correposnd
function calc_theta_bar_i(obs1, obs2,Corpus1, Corpus2, i, γ, Alpha, Elog_Theta, Elog_B1,Elog_B2, count_params, phi1, phi2, w_in_phi_1, w_in_phi_2)

	Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	doc1 = deepcopy(obs1[i])
	doc2 = deepcopy(obs2[i])
	corp1 = deepcopy(Corpus1.Data[i])
	corp2 = deepcopy(Corpus2.Data[i])

	sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
	sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
	#############
	for _u in 1:10
		sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
		for val in unique(corp1[w_in_phi_1[i]])
			x = findall(x -> x == val, corp1[w_in_phi_1[i]])

			y = optimize_phi1_iw_2(Elog_Theta[i,:,:],Elog_B1, count_params,val)
			for xx in x
				phi1[i][xx,:, :] .= y
			end

			sum_phi_1_i .+= length(x).*y
		end
		sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
		for val in unique(corp2[w_in_phi_2[i]])
			x = findall(x -> x == val, corp2[w_in_phi_2[i]])
			y = optimize_phi2_iw_2(Elog_Theta[i,:,:],Elog_B2, count_params,val)
			for xx in x
				phi2[i][xx,:, :] .= y
			end
			sum_phi_2_i .+= length(x).*y
		end
		optimize_γi_2!(count_params.K1, count_params.K2, Alpha,γ[i], sum_phi_1_i, sum_phi_2_i)
		Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	end
	#############
	# for _u in 1:5
	# 		###questionable indexes
	# 	for w in w_in_phi_1[i]
	# 		phi1[i][w,:, :] .= optimize_phi1_iw(phi1[i], Elog_Theta[i,:,:],Elog_B1, count_params, w, corp1)
	# 	end
	# 	for w in w_in_phi_2[i]
	# 		phi2[i][w,:, :] .= optimize_phi2_iw(phi2[i], Elog_Theta[i,:,:],Elog_B2, count_params, w, corp2)
	# 	end
	#
	# 	optimize_γi!(count_params.K1, count_params.K2, Alpha,γ[i], phi1[i], phi2[i])
	# 	Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	# end
		#γ_old[i] .= deepcopy(γ[i])
	theta_bar = γ[i][:,:] ./ sum(γ[i])
	return theta_bar
end


function calc_perp(obs1, obs2,ho1, ho2,corpus1, corpus2, γ, Alpha, Elog_Theta,
 Elog_B1,Elog_B2, count_params, phi1, phi2, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2, B1_est, B2_est)
	corp1 = deepcopy(corpus1)
	corp2 = deepcopy(corpus2)
	l1 = 0.0
	l2 = 0.0
	for i in collect(keys(ho1))

		theta_bar = calc_theta_bar_i(obs1, obs2,corp1, corp2, i, γ, Alpha, Elog_Theta,
		 Elog_B1,Elog_B2, count_params, phi1, phi2, w_in_phi_1, w_in_phi_2)


		for w in w_in_ho_1[i]
			v = corp1.Data[i][w]
			tmp = 0.0
			for k in 1:count_params.K1
				tmp += ((B1_est[k,v]*sum(theta_bar, dims=2)[k,1]))
			end
			l1 += log(tmp)
		end
		for w in w_in_ho_2[i]
			v = corp2.Data[i][w]
			tmp = 0.0
			for k in 1:count_params.K2
				tmp += ((B2_est[k,v]*sum(theta_bar, dims=1)[1,k]))
			end
			l2 += log(tmp)
		end

	end
	l1/= sum(length.(collect(values(ho1))))
	l2/= sum(length.(collect(values(ho2))))

	return exp(-l1), exp(-l2)
end
