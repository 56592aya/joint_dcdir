function init_params(K1::Int64,K2::Int64, beta1_prior_, beta2_prior_,
	 alpha_prior_, corpus1_, corpus2_)
	N = max(corpus1_.N, corpus2_.N)
	alpha_vec = rand(Uniform(0.0,alpha_prior_), (K1*K2))
	# alpha_vec ./=sum(alpha_vec)
	Alpha =  permutedims(reshape(alpha_vec, (K2, K1)), (2,1))
	B1 = rand(Uniform(0.0, beta1_prior_), (K1, corpus1_.V))
	# for k in 1:size(B1,1)
	# 	B1[k,:] ./= sum(B1[k,:])
	# end
	B2 = rand(Uniform(0.0, beta2_prior_), (K2, corpus2_.V))
	# for k in 1:size(B2,1)
	# 	B2[k,:] ./= sum(B2[k,:])
	# end
	γ = [ones(Float64, (K1, K2)) for i in 1:N]
	b1 = deepcopy(B1)
	b2 = deepcopy(B2)
	Elog_B1 = zeros(Float64, (K1, corpus1_.V))
	Elog_B2 = zeros(Float64, (K2, corpus2_.V))
	Elog_Theta = [zeros(Float64, (K1, K2)) for i in 1:N]
	return 	Alpha,B1,B2,Elog_B1,Elog_B2,Elog_Theta,γ,b1,b2
end

function epoch_batches(N::Int64, mb_size::Int64, h_map::Vector{Bool})
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

# function setup_train_test(h::Float64,N::Int64, corp1::Corpus, corp2::Corpus)
#
#
# 	h_count = convert(Int64, floor(h*N))
#
# 	###This should be done so much that we do not leave out vocabulary
# 	while true
# 		# to show whether a doc is in test
# 		# and indices are same as the truth
# 		# and not select in mb or train parameters
# 		h_map = repeat([false], N)
# 		indx = sample(1:N, h_count, replace=false, ordered=true)
# 		h_map[indx] .= true
# 		x1 = [corp1.docs[i].terms for i in collect(1:N)[.!h_map]]
# 		x2 = [corp2.docs[i].terms for i in collect(1:N)[.!h_map]]
#
# 		if (length(unique(vcat(x1...))) == corp1.V) && (length(unique(vcat(x2...))) == corp2.V)
# 			return h_map
# 			# length(unique(vcat(x1...)))
# 			# corp1
# 			# length(unique(vcat(x2...)))
#
# 			# corp1.V
# 		end
# 	end
# end
#
#
# function create_test(h_map, corp1, corp2)
# 	test_ids = findall(h_map)
# 	test_docs_obs_1 = Dict{Int64, Vector{Int64}}()
# 	test_docs_ho_1 = Dict{Int64, Vector{Int64}}()
#
# 	for i in test_ids
# 		if !haskey(test_docs_obs_1, i)
# 			test_docs_obs_1[i] = getkey(test_docs_obs_1, i, Int64[])
# 			test_docs_ho_1[i] = getkey(test_docs_ho_1, i, Int64[])
# 		end
# 		uniqs1 = unique(corp1.docs[i].terms)
# 		uho1 = uniqs1[1:div(length(uniqs1),10)]
# 		uobs1 = uniqs1[div(length(uniqs1),10)+1:end]
# 		doc1 = deepcopy(corp1.docs[i].terms)
# 		w_obs1 = [i for i in doc1 if i in uobs1]
# 		w_ho1 = [i for i in doc1 if i in uho1]
# 		test_docs_obs_1[i] = w_obs1
# 		test_docs_ho_1[i] = w_ho1
# 	end
# 	test_docs_obs_2 = Dict{Int64, Vector{Int64}}()
# 	test_docs_ho_2 = Dict{Int64, Vector{Int64}}()
#
# 	for i in test_ids
# 		if !haskey(test_docs_obs_2, i)
# 			test_docs_obs_2[i] = getkey(test_docs_obs_2, i, Int64[])
# 			test_docs_ho_2[i] = getkey(test_docs_ho_2, i, Int64[])
# 		end
# 		uniqs2 = unique(corp2.docs[i].terms)
# 		uho2 = uniqs2[1:div(length(uniqs2),10)]
# 		uobs2 = uniqs2[div(length(uniqs2),10)+1:end]
# 		doc2 = deepcopy(corp2.docs[i].terms)
# 		w_obs2 = [i for i in doc2 if i in uobs2]
# 		w_ho2 = [i for i in doc2 if i in uho2]
# 		test_docs_obs_2[i] = w_obs2
# 		test_docs_ho_2[i] = w_ho2
# 	end
# 	w_in_phi_1 = Dict{Int64, Array{Int64,1}}()
# 	w_in_phi_2 = Dict{Int64, Array{Int64,1}}()
# 	for i in collect(keys(test_docs_obs_1))
# 		if !haskey(w_in_phi_1, i)
# 			w_in_phi_1[i] = getkey(w_in_phi_1, i, Int64[])
# 		end
# 		for ww in test_docs_obs_1[i]
# 			l = findall(corp1.docs[i].terms .==  ww)
# 			w_in_phi_1[i] = unique(Int64.(vcat(w_in_phi_1[i], l)))
# 		end
# 		w_in_phi_1[i] = sort(unique(w_in_phi_1[i]))
# 	end
# 	for i in collect(keys(test_docs_obs_2))
# 		if !haskey(w_in_phi_2, i)
# 			w_in_phi_2[i] = getkey(w_in_phi_2, i, Int64[])
# 		end
# 		for ww in test_docs_obs_2[i]
# 			l = findall(corp2.docs[i].terms .==  ww)
# 			for el in l
# 				w_in_phi_2[i] = unique(Int64.(vcat(w_in_phi_2[i], l...)))
# 			end
# 		end
# 		w_in_phi_2[i] = sort(unique(w_in_phi_2[i]))
# 	end
# 	w_in_ho_1 = Dict{Int64, Array{Int64,1}}()
# 	w_in_ho_2 = Dict{Int64, Array{Int64,1}}()
# 	for i in collect(keys(test_docs_ho_1))
# 		if !haskey(w_in_ho_1, i)
# 			w_in_ho_1[i] = getkey(w_in_ho_1, i, Int64[])
# 		end
# 		for ww in test_docs_ho_1[i]
# 			l = findall(corp1.docs[i].terms .==  ww)
# 			w_in_ho_1[i] = unique(Int64.(vcat(w_in_ho_1[i], l)))
# 		end
# 		w_in_ho_1[i] = sort(unique(w_in_ho_1[i]))
# 	end
# 	for i in collect(keys(test_docs_ho_2))
# 		if !haskey(w_in_ho_2, i)
# 			w_in_ho_2[i] = getkey(w_in_ho_2, i, Int64[])
# 		end
# 		for ww in test_docs_ho_2[i]
# 			l = findall(corp2.docs[i].terms .==  ww)
# 			w_in_ho_2[i] = unique(Int64.(vcat(w_in_ho_2[i], l)))
# 		end
# 		w_in_ho_2[i] = sort(unique(w_in_ho_2[i]))
# 	end
#
# 	return test_docs_ho_1, test_docs_obs_1, test_docs_ho_2, test_docs_obs_2, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2
# end


function setup_hmap(model, h)
	h_count = convert(Int64, floor(h*N))
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	while true
		h_map = repeat([false], N)
		inds = sample(1:N, h_count, replace=false, ordered=true)
		h_map[inds] .= true
		x1 = [corp1.docs[i].terms for i in collect(1:corp1.N)[.!h_map]]
		x2 = [corp2.docs[i].terms for i in collect(1:corp2.N)[.!h_map]]
		cond1 = any(.!isempty.(x1)) &&  any(.!isempty.(x2))
		cond2 = (length(unique(vcat(x1...))) == corp1.V) && (length(unique(vcat(x2...))) == corp2.V)
		if cond1 & cond2
			return h_map
		end
	end
end

function split_ho_obs(model, h_map)
	test_ids = findall(h_map)
	hos1_dict = Dict{Int64, Vector{Int64}}()
	obs1_dict = Dict{Int64, Vector{Int64}}()
	hos2_dict = Dict{Int64, Vector{Int64}}()
	obs2_dict = Dict{Int64, Vector{Int64}}()
	for i in test_ids
		if !haskey(hos1_dict, i)
			hos1_dict[i] = getkey(hos1_dict, i, Int64[])
			obs1_dict[i] = getkey(obs1_dict, i, Int64[])
			hos2_dict[i] = getkey(hos2_dict, i, Int64[])
			obs2_dict[i] = getkey(obs2_dict, i, Int64[])
		end
		terms_1 = model.Corpus1.docs[i].terms
		terms_2 = model.Corpus2.docs[i].terms
		partition_1 = div(length(terms_1),5)
		partition_2 = div(length(terms_2),5)
		hos1  = terms_1[1:partition_1]
		obs1  = terms_1[partition_1+1:end]
		hos2  = terms_2[1:partition_2]
		obs2  = terms_2[partition_2+1:end]
		hos1_dict[i] = hos1
		obs1_dict[i] = obs1
		hos2_dict[i] = hos2
		obs2_dict[i] = obs2
	end

	return hos1_dict,obs1_dict,hos2_dict,obs2_dict
end
