# function init_params(K1::Int64,K2::Int64, beta1_prior_, beta2_prior_,
# 	 alpha_prior_, corpus1_, corpus2_)
# 	N = max(corpus1_.N, corpus2_.N)
# 	alpha_vec = rand(Uniform(alpha_prior_/2,alpha_prior_*2), (K1*K2)) .* ones(Float64, K1*K2)
# 	Alpha =  permutedims(reshape(alpha_vec, (K2, K1)), (2,1))
# 	beta1 = ones(Float64, (K1,corpus1_.V)) .* rand(Uniform(beta1_prior_/4, beta1_prior_*2), K1)
# 	beta2 = ones(Float64, (K2,corpus2_.V)) .* rand(Uniform(beta2_prior_/4, beta2_prior_*2), K2)
# 	#variational params
#
# 	phi1 = [(1.0/(K1*K2)) .* ones(Float64, (corpus1_.doc_lens[i], K1, K2)) for i in 1:N]
# 	phi2 = [(1.0/(K1*K2)) .* ones(Float64, (corpus2_.doc_lens[i], K1, K2)) for i in 1:N]
#
# 	γ = [ones(Float64, (K1, K2)) for i in 1:N]
# 	b1 = deepcopy(beta1)
# 	b2 = deepcopy(beta2)
# 	Elog_B1 = zeros(Float64, (K1, corpus1_.V))
# 	Elog_B2 = zeros(Float64, (K2, corpus2_.V))
# 	Elog_Theta = zeros(Float64, (N, K1, K2))
# 	## Also make sure vocab is the ones used.
# 	return 	alpha_vec, Alpha,beta1, beta2,
# 			phi1, phi2, γ, b1, b2,Elog_B1, Elog_B2,Elog_Theta
# end
function ctm(C::Corpus)
	d = Dict{Int64, Int64}()
	for i in 1:C.N
		for w in C.Data[i]
			if !haskey(d, w)
				d[w] = getkey(d, w, 1)
			else
				d[w] += 1
			end
		end
	end
	return d
end
function dtm(C::Corpus)
	d = zeros(Int64, (C.N, C.V))
	for i in 1:C.N
		for w in C.Data[i]
			d[i,w] += 1
		end
	end
	return d
end



function valid_holdout_selection(dtm, mincount)
	arraylist = [Int64[] for i in 1:size(dtm, 1)]
	for i in 1:size(dtm, 1)
		counts = dtm[i,findall(dtm[i,:] .> mincount)] .- mincount
		tmp_arr =  findall(dtm[i,:] .> mincount)
		for j in 1:length(counts)
			arraylist[i] = vcat(arraylist[i], repeat([tmp_arr[j]], counts[j]))
		end
	end
	return arraylist
end


function train_holdout(h::Float64,corp::Corpus, select_from::Array{Array{Int64,1},1},
	 dtm::Matrix{Int64})
	#should choose from words that have at least
	h = 0.10
	h_count = convert(Int64, floor(h*sum(dtm)))

	count = 0
	select = deepcopy(select_from)
	copy_corp = deepcopy(corp)
	copy_dtm = deepcopy(dtm)

	tmp_holdout = [Int64[] for i in 1:copy_corp.N]
	i=1
	while count < h_count
		# global i, count, h_count
		# dat = copy_corp.Data[i]
		if isempty(select[i])
			i += 1
			if i > copy_corp.N
				i=1
			end
			continue
		end
		el = sample(select[i], 1)[1]
		tmp_holdout[i] = vcat(tmp_holdout[i], el)
		count += 1
		deleteat!(select[i], findfirst(select[i] .== el))
		deleteat!(copy_corp.Data[i], findfirst(copy_corp.Data[i] .== el))
		copy_corp.doc_lens[i] -= 1

		i +=1

		if i > copy_corp.N
			i = 1
		end
		# if count % 50 == 0
		# 	println(count)
		# end


	end
	return (copy_corp, tmp_holdout)
end
function setup_train_test(h::Float64,N::Int64, corp1::Corpus, corp2::Corpus)

	##remove this line later
	# h = 0.01
	h_count = convert(Int64, floor(h*N))

	###This should be done so much that we do not leave out vocabulary
	while true
		# to show whether a doc is in test
		# and indices are same as the truth
		# and not select in mb or train parameters
		h_map = repeat([false], N)
		indx = sample(1:N, h_count, replace=false, ordered=true)
		h_map[indx] .= true
		x1 = [corp1.docs[i].terms for i in collect(1:N)[.!h_map]]
		x2 = [corp2.docs[i].terms for i in collect(1:N)[.!h_map]]

		if (length(unique(vcat(x1...))) == corp1.V) && (length(unique(vcat(x2...))) == corp2.V)
			return h_map
			# length(unique(vcat(x1...)))
			# corp1
			# length(unique(vcat(x2...)))

			# corp1.V
		end
	end
end


function create_test(h_map, corp1, corp2)
	test_ids = findall(h_map)
	test_docs_obs_1 = Dict{Int64, Vector{Int64}}()
	test_docs_ho_1 = Dict{Int64, Vector{Int64}}()

	for i in test_ids
		if !haskey(test_docs_obs_1, i)
			test_docs_obs_1[i] = getkey(test_docs_obs_1, i, Int64[])
			test_docs_ho_1[i] = getkey(test_docs_ho_1, i, Int64[])
		end
		uniqs1 = unique(corp1.docs[i].terms)
		uho1 = uniqs1[1:div(length(uniqs1),10)]
		uobs1 = uniqs1[div(length(uniqs1),10)+1:end]
		doc1 = deepcopy(corp1.docs[i].terms)
		w_obs1 = [i for i in doc1 if i in uobs1]
		w_ho1 = [i for i in doc1 if i in uho1]
		test_docs_obs_1[i] = w_obs1
		test_docs_ho_1[i] = w_ho1
	end
	test_docs_obs_2 = Dict{Int64, Vector{Int64}}()
	test_docs_ho_2 = Dict{Int64, Vector{Int64}}()

	for i in test_ids
		if !haskey(test_docs_obs_2, i)
			test_docs_obs_2[i] = getkey(test_docs_obs_2, i, Int64[])
			test_docs_ho_2[i] = getkey(test_docs_ho_2, i, Int64[])
		end
		uniqs2 = unique(corp2.docs[i].terms)
		uho2 = uniqs2[1:div(length(uniqs2),10)]
		uobs2 = uniqs2[div(length(uniqs2),10)+1:end]
		doc2 = deepcopy(corp2.docs[i].terms)
		w_obs2 = [i for i in doc2 if i in uobs2]
		w_ho2 = [i for i in doc2 if i in uho2]
		test_docs_obs_2[i] = w_obs2
		test_docs_ho_2[i] = w_ho2
	end
	w_in_phi_1 = Dict{Int64, Array{Int64,1}}()
	w_in_phi_2 = Dict{Int64, Array{Int64,1}}()
	for i in collect(keys(test_docs_obs_1))
		if !haskey(w_in_phi_1, i)
			w_in_phi_1[i] = getkey(w_in_phi_1, i, Int64[])
		end
		for ww in test_docs_obs_1[i]
			l = findall(corp1.docs[i].terms .==  ww)
			w_in_phi_1[i] = unique(Int64.(vcat(w_in_phi_1[i], l)))
		end
		w_in_phi_1[i] = sort(unique(w_in_phi_1[i]))
	end
	for i in collect(keys(test_docs_obs_2))
		if !haskey(w_in_phi_2, i)
			w_in_phi_2[i] = getkey(w_in_phi_2, i, Int64[])
		end
		for ww in test_docs_obs_2[i]
			l = findall(corp2.docs[i].terms .==  ww)
			for el in l
				w_in_phi_2[i] = unique(Int64.(vcat(w_in_phi_2[i], l...)))
			end
		end
		w_in_phi_2[i] = sort(unique(w_in_phi_2[i]))
	end
	w_in_ho_1 = Dict{Int64, Array{Int64,1}}()
	w_in_ho_2 = Dict{Int64, Array{Int64,1}}()
	for i in collect(keys(test_docs_ho_1))
		if !haskey(w_in_ho_1, i)
			w_in_ho_1[i] = getkey(w_in_ho_1, i, Int64[])
		end
		for ww in test_docs_ho_1[i]
			l = findall(corp1.docs[i].terms .==  ww)
			w_in_ho_1[i] = unique(Int64.(vcat(w_in_ho_1[i], l)))
		end
		w_in_ho_1[i] = sort(unique(w_in_ho_1[i]))
	end
	for i in collect(keys(test_docs_ho_2))
		if !haskey(w_in_ho_2, i)
			w_in_ho_2[i] = getkey(w_in_ho_2, i, Int64[])
		end
		for ww in test_docs_ho_2[i]
			l = findall(corp2.docs[i].terms .==  ww)
			w_in_ho_2[i] = unique(Int64.(vcat(w_in_ho_2[i], l)))
		end
		w_in_ho_2[i] = sort(unique(w_in_ho_2[i]))
	end

	return test_docs_ho_1, test_docs_obs_1, test_docs_ho_2, test_docs_obs_2, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2
end
# sum([sum(length(holdout_1[i])) for i in 1:2000])
# sum([sum(length(holdout_2[i])) for i in 1:2000])
