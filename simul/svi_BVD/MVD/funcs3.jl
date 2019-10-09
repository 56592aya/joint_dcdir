function update_Elogtheta!(Elog_, γ_)
	Elog_ .= Elog.(γ_)
end
function update_Elogtheta_i!(Elog_::Matrix{Float64},γ_::Matrix{Float64})
	Elog_ .= Elog(γ_)
end
function update_Elogtheta_i!(model, i)
	model.Elog_Theta[i] .= deepcopy(Elog(model.γ[i]))
end
function update_Elogb!(Elog_,b_)
	for k in 1:size(Elog_,1)
		Elog_[k,:] .=  Elog(b_[k,:])
	end
end
function estimate_thetas(gamma)
	theta_est = deepcopy(gamma)
	for i in 1:length(theta_est)
		s = sum(gamma[i])
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

# function optimize_γ!(mb_, mindex_, K1_, K2_, Alpha_,γ_, phi1_, phi2_)
# 	for i in mb_[mindex_]
# 		for k1 in 1:K1_
# 			for k2 in 1:K2_
# 				γ_[i][k1, k2] = Alpha_[k1 ,k2] +sum(phi1_[i][:,k1, k2])+ sum(phi2_[i][:,k1, k2])
# 			end
# 		end
# 	end
# end
###

# function optimize_γi!(K1_::Int64, K2_::Int64, Alpha_::Matrix{Float64},γ_::Matrix{Float64}, sum_phi1_::Matrix{Float64}, sum_phi2_::Matrix{Float64})
# 	@.(γ_ = Alpha_ + sum_phi1_ + sum_phi2_)
#
# end
function optimize_γi!(model::MVD, i,sum_phi1_::Matrix{Float64}, sum_phi2_::Matrix{Float64})
	model.γ[i] .= deepcopy(model.Alpha .+ sum_phi1_ .+ sum_phi2_)
end
function optimize_b(len_mb, model_B, sum_phi_mb,count_params)
	b_ = deepcopy(model_B)
	b_ .+= (count_params.N/len_mb) .* sum_phi_mb
	return b_
end

function optimize_phi_iw(model::MVD, i, mode::Int64, v)
	#####
	if mode == 1
		return softmax(model.Elog_Theta[i] .+ model.Elog_B1[:,v])
	else
		return softmax(model.Elog_Theta[i] .+ model.Elog_B2[:,v]')
	end
end




##### w , and v needs to be fixed in holdout ho and obs what to save probably their indices that correposnd
function calc_theta_bar_i(obs1_dict, obs2_dict, i, model, count_params)
	update_Elogtheta_i!(model, i)
	doc1 = deepcopy(obs1_dict[i])
	doc2 = deepcopy(obs2_dict[i])
	corp1 = deepcopy(model.Corpus1.docs[i])
	corp2 = deepcopy(model.Corpus2.docs[i])
	sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
	sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
	#############
	for _u in 1:100
		sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
		obs_words_corp1inds = [find_all(d,corp1.terms)[1] for d in doc1]
		for (key,val) in enumerate(corp1.terms[obs_words_corp1inds])
			y = optimize_phi_iw(model, i,1,val)
			sum_phi_1_i .+= corp1.counts[key].*y
		end
		sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
		obs_words_corp2inds = [find_all(d,corp2.terms)[1] for d in doc2]
		for (key,val) in enumerate(corp2.terms[obs_words_corp2inds])
			y = optimize_phi_iw(model, i, 2, val)
			sum_phi_2_i .+= corp2.counts[key].*y
		end
		optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
		update_Elogtheta_i!(model, i)
	end
	theta_bar = model.γ[i][:,:] ./ sum(model.γ[i])
	return theta_bar
end


function calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,count_params, B1_est, B2_est)
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	l1 = 0.0
	l2 = 0.0
	for i in collect(keys(hos1_dict))
		theta_bar = calc_theta_bar_i(obs1_dict, obs2_dict,i, model, count_params)
		for v in hos1_dict[i]
			tmp = 0.0
			for k in 1:count_params.K1
				tmp += ((B1_est[k,v]*sum(theta_bar, dims=2)[k,1]))
			end
			l1 += log(tmp)
		end
		for v in hos2_dict[i]
			tmp = 0.0
			for k in 1:count_params.K2
				tmp += ((B2_est[k,v]*sum(theta_bar, dims=1)[1,k]))
			end
			l2 += log(tmp)
		end

	end
	l1/= sum(length.(collect(values(hos1_dict))))
	l2/= sum(length.(collect(values(hos2_dict))))

	return exp(-l1), exp(-l2)
end

function update_alpha!(model, mb, rate_)
	g = zeros(Float64, prod(size(model.Alpha)))
	N = length(mb)
	init_ = ones(Float64, size(model.Alpha))
	# g = N.*(vectorize_mat(-Elog(model.Alpha)) + (sum(vectorize_mat.(Elog.(model.γ[mb])))./N))
	# c = N*trigamma_(sum(model.Alpha))
	# q = -N.*trigamma_.(vectorize_mat(model.Alpha))
	# rate_ = 0.5

	for _ in 1:100

		g = N.*(vectorize_mat(-Elog(init_)) + (sum(vectorize_mat.(Elog.(model.γ[mb])))./N))
		c = N*trigamma_(sum(init_))
		q = -N.*trigamma_.(vectorize_mat(init_))
		b = sum(g./q)/(1.0/c+sum(1.0./q))
		dalpha = -(g .- b)./q
		init_ .+= matricize_vec(rate_.*dalpha, model.K1, model.K2)
		if any(init_ .<= 0.0)
			rate_ /=sqrt(2.0)
			init_ = ones(Float64, size(model.Alpha))
		end
		# norm(g)
		if norm(g) < 1e-5
			model.Alpha .= init_[:,:]
			break
		end
		# if all(rate_*dalpha + vectorize_mat(model.Alpha) .> 0)
		# 	model.Alpha += matricize_vec(rate_.*dalpha, model.K1, model.K2)
		# else
		# 	return
		# end
		model.Alpha .= init_[:,:]
	end
end
