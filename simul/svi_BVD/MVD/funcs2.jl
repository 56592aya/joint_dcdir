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

function optimize_γi!(model::MVD, i,sum_phi1_::Matrix{Float64}, sum_phi2_::Matrix{Float64})
	model.γ[i] .= deepcopy(model.Alpha .+ sum_phi1_ .+ sum_phi2_)
end
function optimize_b(len_mb, model_beta, sum_phi_mb,count_params)
	b_ = deepcopy(model_beta)
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
			sum_phi_1_i .+= corp1.counts[key].* y
		end
		sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
		obs_words_corp2inds = [find_all(d,corp2.terms)[1] for d in doc2]
		for (key,val) in enumerate(corp2.terms[obs_words_corp2inds])
			y = optimize_phi_iw(model, i, 2, val)
			sum_phi_2_i .+= corp2.counts[key].* y
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


# function update_phis_gammas!(model, i, sum_phi_1_i, sum_phi_2_i,sum_phi_1_mb,sum_phi_2_mb,zeroer_i,doc1,doc2,gamma_c,γ_old)
#
# 	for _u in 1:50
# 		if rand()>= .5
# 			copyto!(sum_phi_1_i, zeroer_i)
# 			for (w,val) in enumerate(doc1.terms)
#
# 				y = optimize_phi_iw(model, i,1,val)
#
# 				sum_phi_1_i .+= doc1.counts[w] .* y
# 				if (_u == 50) || gamma_c
# 					sum_phi_1_mb[:,val] .+= sum(doc1.counts[w].* y, dims = 2)[:,1]
# 				end
# 			end
# 			copyto!(sum_phi_2_i, zeroer_i)
# 			for (w,val) in enumerate(doc2.terms)
# 				y = optimize_phi_iw(model, i,2,val)
# 				sum_phi_2_i .+= doc2.counts[w] .* y
# 				if (_u == 50) || gamma_c
# 					sum_phi_2_mb[:,val] .+= sum(doc2.counts[w] .* y , dims = 1)[1,:]
# 				end
# 			end
# 			optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
# 			update_Elogtheta_i!(model,i)
# 			# println(mean(abs.(γ_old .- model.γ[i])./model.γ[i]))
# 			if (mean(abs.(γ_old .- model.γ[i])./model.γ[i])) < 1e-4
# 				gamma_c = true
# 			end
# 			γ_old = deepcopy(model.γ[i])
# 		else
# 			copyto!(sum_phi_2_i, zeroer_i)
# 			for (w,val) in enumerate(doc2.terms)
# 				y = optimize_phi_iw(model, i,2,val)
# 				sum_phi_2_i .+= doc2.counts[w] .* y
# 				if (_u == 50) || gamma_c
# 					sum_phi_2_mb[:,val] .+= sum(doc2.counts[w] .* y , dims = 1)[1,:]
# 				end
# 			end
# 			copyto!(sum_phi_1_i, zeroer_i)
# 			for (w,val) in enumerate(doc1.terms)
#
# 				y = optimize_phi_iw(model, i,1,val)
#
# 				sum_phi_1_i .+= doc1.counts[w] .* y
# 				if (_u == 50) || gamma_c
# 					sum_phi_1_mb[:,val] .+= sum(doc1.counts[w].* y, dims = 2)[:,1]
# 				end
# 			end
# 			optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
# 			update_Elogtheta_i!(model,i)
# 			# println(mean(abs.(γ_old .- model.γ[i])./model.γ[i]))
# 			if (mean(abs.(γ_old .- model.γ[i])./model.γ[i])) < 1e-4
# 				gamma_c = true
# 			end
# 			γ_old = deepcopy(model.γ[i])
# 		end
# 	end
# end



function update_phis_gammas!(model, i, sum_phi_1_i, sum_phi_2_i,sum_phi_1_mb,sum_phi_2_mb,zeroer_i,doc1,doc2,gamma_c,γ_old)
	MAXLOOP = 50
	counter  = 0
	old_change = 1000000.0
	gamma_change = 500.0
	while !(gamma_c) || counter <= MAXLOOP
		# global counter, MAXLOOP,old_change, gamma_change
		copyto!(sum_phi_1_i, zeroer_i)
		copyto!(sum_phi_2_i, zeroer_i)
		for (w,val) in enumerate(doc1.terms)
			y = optimize_phi_iw(model, i,1,val)
			sum_phi_1_i .+= doc1.counts[w] .* y
		end
		for (w,val) in enumerate(doc2.terms)
			y = optimize_phi_iw(model, i,2,val)
			sum_phi_2_i .+= doc2.counts[w] .* y
		end
		optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
		update_Elogtheta_i!(model,i)
		gamma_change = mean(abs.(γ_old .- model.γ[i])./model.γ[i])

		# println(gamma_change)
		# println(old_change)
		# println(gamma_change > old_change)
		if (gamma_change) < 1e-5 || gamma_change > old_change
			old_change = gamma_change
			gamma_c = true
			copyto!(sum_phi_1_i, zeroer_i)
			copyto!(sum_phi_2_i, zeroer_i)
			for (w,val) in enumerate(doc1.terms)
				y = optimize_phi_iw(model, i,1,val)
				sum_phi_1_i .+= doc1.counts[w] .* y
				sum_phi_1_mb[:,val] .+= sum(doc1.counts[w].* y, dims = 2)[:,1]
			end
			for (w,val) in enumerate(doc2.terms)
				y = optimize_phi_iw(model, i,2,val)
				sum_phi_2_i .+= doc2.counts[w] .* y
				sum_phi_2_mb[:,val] .+= sum(doc2.counts[w] .* y , dims = 1)[1,:]
			end
			optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
			update_Elogtheta_i!(model,i)
		end
		old_change = gamma_change

		γ_old = deepcopy(model.γ[i])
		if counter == MAXLOOP
			gamma_c = true
		end
		counter += 1
	end
end


function update_beta1!(model,k, rate_)
	init_ = deepcopy(model.B1[k,:])
	# init_ ./= sum(init_)
	for _ in 1:500
		g = digamma_(sum(init_)) .- digamma_.(init_) + Elog(model.b1[k,:])
		q = -trigamma_.(init_)
		c = trigamma_(sum(init_))
		b = sum(g./q) / (1.0/c + sum(1.0./q))
		dbeta = -(g.-b)./q
		init_ .+= rate_.*dbeta
		# init_ ./= sum(init_)
		if any(init_ .<= 0.0)
			rate_ /=sqrt(2.0)
			init_ = deepcopy(model.B1[k,:])
		end
		# println(norm(g))
		if norm(g) < 1e-5
			model.B1[k,:] = deepcopy(init_)
			break
		end
		model.B1[k,:] = deepcopy(init_)
	end
end

function update_beta2!(model,k, rate_)
	# g = zeros(Float64, size(model.B2,2))
	init_ = deepcopy(model.B2[k,:])
	# init_ ./=sum(init_)
	# Hinv = zeros(Float64, (size(model.B2,2),size(model.B2,2)))

	for _ in 1:500
		g = digamma_(sum(init_)) .- digamma_.(init_) + Elog(model.b2[k,:])
		q = -trigamma_.(init_)
		c = trigamma_(sum(init_))
		b = sum(g./q) / (1.0/c + sum(1.0./q))
		dbeta = -(g.-b)./q
		init_ .+= rate_.*dbeta
		# init_ ./= sum(init_)
		if any(init_ .<= 0.0)
			rate_ /=sqrt(2.0)
			init_ = deepcopy(model.B2[k,:])
		end
		if norm(g) < 1e-5
			model.B2[k,:] = deepcopy(init_)
			break
		end
		model.B2[k,:] = deepcopy(init_)
	end
end
function update_alpha!(model, mb, rate_)
	# g = zeros(Float64, prod(size(model.Alpha)))
	N = length(mb)
	init_ = deepcopy(model.Alpha)

	for _ in 1:500

		g = N.*(vectorize_mat(-Elog(init_)) + (sum(vectorize_mat.(Elog.(model.γ[mb])))./N))
		c = N*trigamma_(sum(init_))
		q = -N.*trigamma_.(vectorize_mat(init_))
		b = sum(g./q)/(1.0/c+sum(1.0./q))
		dalpha = -(g .- b)./q
		init_ .+= matricize_vec(rate_.*dalpha, model.K1, model.K2)
		# init_ ./= sum(init_)
		if any(init_ .<= 0.0)
			rate_ /=sqrt(2.0)
			init_ = deepcopy(model.Alpha)
		end
		if norm(g) < 1e-5
			model.Alpha .= init_[:,:]
			break
		end
		model.Alpha .= init_[:,:]
	end
end
