# using BenchmarkTools
include("dgp.jl")
include("init.jl")


function epoch_batches(N::Int64, mb_size::Int64)
	# mb_size=64
	div_ = div(N, mb_size)
	nb = (div_ * mb_size - N) < 0 ? div_ + 1 : div_
	y = shuffle(collect(1:N))
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

function get_lr(iter, S, κ)
	return (S+iter)^(-κ)
end
