
function update_Elogtheta!(Elog_, γ_)
	Elog_ .= Elog.(γ_)
end

function update_Elogtheta_i!(model, i)
	model.Elog_Theta[i] .= Elog(model.γ[i])
end

function update_Elogb!(model, mode)
	if mode == 1
		for (k,row) in enumerate(eachrow(model.b1))
			model.Elog_B1[k,:].= Elog(row)
		end
	else
		for (k,row) in enumerate(eachrow(model.b2))
			model.Elog_B2[k,:] .=  Elog(row)
		end
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

function optimize_γi!(model::MVD, i)
	@. (model.γ[i]= model.Alpha + model.sum_phi_1_i + model.sum_phi_2_i)
end

function optimize_b!(b_,len_mb, model_beta, sum_phi_mb,count_params)
	copyto!(b_, model_beta)
	@.(b_ += (count_params.N/len_mb) * sum_phi_mb)
end


function optimize_phi_iw!(model::MVD, i, mode::Int64, v)
	#####
	if mode == 1
		softmax!(model.temp, @.(model.Elog_Theta[i] + model.Elog_B1[:,v]))
	else
		softmax!(model.temp, @.(model.Elog_Theta[i] + model.Elog_B2[:,v]'))
	end
end


function update_phis_gammas!(model, i,zeroer_i,doc1,doc2,gamma_c)
	MAXLOOP = 50
	counter  = 0
	old_change = 1000000.0
	gamma_change = 500.0

	while !(gamma_c) || counter <= MAXLOOP
		# global counter, MAXLOOP,old_change, gamma_change
		copyto!(model.sum_phi_1_i, zeroer_i)
		copyto!(model.sum_phi_2_i, zeroer_i)
		# copyto!(temp,zeroer_i)
		for (w,val) in enumerate(doc1.terms)
			optimize_phi_iw!(model, i,1,val)
			@. model.sstat_i = doc1.counts[w] * model.temp
			@.(model.sum_phi_1_i += model.sstat_i)
		end
		for (w,val) in enumerate(doc2.terms)
			optimize_phi_iw!(model, i,2,val)
			@. model.sstat_i = doc2.counts[w] * model.temp
			@.(model.sum_phi_2_i += model.sstat_i)
		end
		optimize_γi!(model, i)
		update_Elogtheta_i!(model,i)
		gamma_change = mean(@.(abs(model.old_γ - model.γ[i])/model.γ[i]))

		# println(gamma_change)
		# println(old_change)
		# println(gamma_change > old_change)
		if ((gamma_change) < 1e-5) || (gamma_change > old_change)
			old_change = gamma_change
			gamma_c = true
			copyto!(model.sum_phi_1_i, zeroer_i)
			copyto!(model.sum_phi_2_i, zeroer_i)
			# copyto!(temp,zeroer_i)

			for (w,val) in enumerate(doc1.terms)
				optimize_phi_iw!(model, i,1,val)
				@. (model.sstat_i = doc1.counts[w] * model.temp)
				@.(model.sum_phi_1_i += model.sstat_i)
				model.sstat_mb_1 .= sum(model.sstat_i, dims = 2)[:,1]
				@.(model.sum_phi_1_mb[:,val] += model.sstat_mb_1)
			end
			for (w,val) in enumerate(doc2.terms)
				optimize_phi_iw!(model, i,2,val)
				@.(model.sstat_i = doc2.counts[w] * model.temp)
				@.(model.sum_phi_2_i += model.sstat_i)
				model.sstat_mb_2 .= sum(model.sstat_i , dims = 1)[1,:]
				@.(model.sum_phi_2_mb[:,val] += model.sstat_mb_2)
			end
			optimize_γi!(model, i)
			update_Elogtheta_i!(model,i)
		end
		old_change = gamma_change

		copyto!(model.old_γ,model.γ[i])
		if counter == MAXLOOP
			gamma_c = true
		end
		counter += 1
	end
end
##### w , and v needs to be fixed in holdout ho and obs what to save probably their indices that correposnd
function calc_theta_bar_i(obs1_dict, obs2_dict, i, model, count_params,zeroer_i)
	update_Elogtheta_i!(model, i)
	doc1 = obs1_dict[i]
	doc2 = obs2_dict[i]
	corp1 = model.Corpus1.docs[i]
	corp2 = model.Corpus2.docs[i]
	copyto!(model.sum_phi_1_i, zeroer_i)
	copyto!(model.sum_phi_2_i, zeroer_i)
	copyto!(model.old_γ ,  model.γ[i])
	MAXLOOP = 100
	counter  = 0
	old_change = 1000000.0
	gamma_change = 500.0
	gamma_c = false
	#############
	while !(gamma_c) || counter <= MAXLOOP

		copyto!(model.sum_phi_1_i, zeroer_i)

		obs_words_corp1inds = [find_all(d,corp1.terms)[1] for d in doc1]
		for (key,val) in enumerate(corp1.terms[obs_words_corp1inds])
			optimize_phi_iw!(model, i,1,val)
			@.(model.sstat_i = corp1.counts[key] * model.temp)
			@.(model.sum_phi_1_i += model.sstat_i)
		end
		copyto!(model.sum_phi_2_i, zeroer_i)
		obs_words_corp2inds = [find_all(d,corp2.terms)[1] for d in doc2]
		for (key,val) in enumerate(corp2.terms[obs_words_corp2inds])
			optimize_phi_iw!(model, i,2,val)
			@.(model.sstat_i = corp2.counts[key] * model.temp)
			@.(model.sum_phi_2_i += model.sstat_i)
		end
		optimize_γi!(model, i)
		update_Elogtheta_i!(model, i)
		gamma_change = mean(abs.(model.old_γ .- model.γ[i])./model.γ[i])
		if ((gamma_change) < 1e-5) || (gamma_change > old_change)
			old_change = gamma_change
			gamma_c = true
		end
		old_change = gamma_change
		counter +=1
	end
	theta_bar = model.γ[i][:,:] ./ sum(model.γ[i])
	return theta_bar
end


function calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,count_params, B1_est, B2_est,zeroer_i)
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	l1 = 0.0
	l2 = 0.0
	for i in collect(keys(hos1_dict))
		theta_bar = calc_theta_bar_i(obs1_dict, obs2_dict,i, model, count_params,zeroer_i)
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
			rate_ *=.9
			init_ = deepcopy(model.B1[k,:])
		end
		# println(norm(g))
		if norm(g) < 1e-2
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
	pg = Elog(model.b2[k,:])
	for _ in 1:500
		g = digamma_(sum(init_)) .- digamma_.(init_) + pg
		q = -trigamma_.(init_)
		c = trigamma_(sum(init_))
		b = sum(g./q) / (1.0/c + sum(1.0./q))
		dbeta = -(g.-b)./q
		init_ .+= rate_.*dbeta
		# init_ ./= sum(init_)
		if any(init_ .<= 0.0)
			rate_ *=.9
			init_ = deepcopy(model.B2[k,:])
		end
		if norm(g) < 1e-2
			model.B2[k,:] = deepcopy(init_)
			break
		end
		model.B2[k,:] = deepcopy(init_)
	end
end
function update_alpha!(model, mb, rate_, count_params)
	# g = zeros(Float64, prod(size(model.Alpha)))
	n = length(mb)

	logphat = sum([Elog(model.γ[i]) for i in mb]) / n


	K = prod(size(model.Alpha))
	X = hcat(vectorize_mat.(model.γ[mb])...)
	temp_ = sum(X, dims=2)[:,1]./K
	temp_ = matricize_vec(temp_, K1, K2)
	init_ = deepcopy(temp_)

	for _ in 1:500
		copyto!(model.old_Alpha, init_)
		alpha0 = sum(init_)
		g = n.*(vectorize_mat(-Elog(init_))) + logphat

		c = N*trigamma_(alpha0)
		q = -N.*trigamma_.(vectorize_mat(init_))
		b = sum(g./q)/(1.0/c+sum(1.0./q))
		dalpha = -(g .- b)./q
		init_ .+= matricize_vec(rate_.*dalpha, model.K1, model.K2)

		if any(init_ .<= 0.0)
			rate_ *=.9
			init_ = deepcopy(temp_)
		end
		if norm(g) < 1e-4
			copyto!(model.Alpha, init_)
			break
		end
		copyto!(model.Alpha, init_)
	end
end
