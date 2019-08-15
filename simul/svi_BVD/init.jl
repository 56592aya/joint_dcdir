function init_params(params_::Params, beta1_prior_, beta2_prior_,
	 alpha_prior_, corpus1_::Corpus, corpus2_::Corpus)
	# N = max(length(corp1_), length(corp2_))
	# K1 = K1_
	# K2 = K2_
	# wlens1 = [length(corp1_[i]) for i in 1:N]
	# wlens2 = [length(corp2_[i]) for i in 1:N]
	#priors
	alpha_vec = rand(Uniform(alpha_prior_/2,alpha_prior_*2), (params_.K1*params_.K2)) .* ones(Float64, params_.K1*params_.K2)
	Alpha =  permutedims(reshape(alpha_vec, (params_.K2, params_.K1)), (2,1))
	beta1 = ones(Float64, (params_.K1,corpus1_.V)) .* rand(Uniform(beta1_prior_/4, beta1_prior_*2), params_.K1)
	beta2 = ones(Float64, (params_.K2,corpus2_.V)) .* rand(Uniform(beta2_prior_/4, beta2_prior_*2), params_.K2)
	#variational params
	phi1 = [(1.0/(params_.K1*params_.K2)) .* ones(Float64, (corpus1_.doc_lens[i], params_.K1, params_.K2)) for i in 1:params_.N]
	phi2 = [(1.0/(params_.K1*params_.K2)) .* ones(Float64, (corpus2_.doc_lens[i], params_.K1, params_.K2)) for i in 1:params_.N]
	γ = [ones(Float64, (params_.K1, params_.K2)) for i in 1:params_.N]
	# for i in 1:N
	# 	# γ[i] = deepcopy(Alpha)
	# 	# γ[i] = [ones(Float64, (K1, K2)) for i in 1:N]
	# end
	b1 = deepcopy(beta1)
	b2 = deepcopy(beta2)
	Elog_B1 = zeros(Float64, (params_.K1, corpus1_.V))
	Elog_B2 = zeros(Float64, (params_.K2, corpus2_.V))
	Elog_Theta = zeros(Float64, (params_.N, params_.K1, params_.K2))
	## Also make sure vocab is the ones used.
	return 	alpha_vec, Alpha,beta1, beta2,
			phi1, phi2, γ, b1, b2,Elog_B1, Elog_B2,Elog_Theta
end
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

# sum([sum(length(holdout_1[i])) for i in 1:2000])
# sum([sum(length(holdout_2[i])) for i in 1:2000])
