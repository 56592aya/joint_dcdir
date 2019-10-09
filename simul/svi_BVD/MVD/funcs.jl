# using BenchmarkTools
include("utils.jl")
include("MVD.jl")
include("dgp.jl")
include("init.jl")


function init_params(K1::Int64,K2::Int64, beta1_prior_, beta2_prior_,
	 alpha_prior_, corpus1_, corpus2_)

	N = max(corpus1_.N, corpus2_.N)

	alpha_vec = rand(Uniform(alpha_prior_/2,alpha_prior_*2), (K1*K2)) .* ones(Float64, K1*K2)
	Alpha =  permutedims(reshape(alpha_vec, (K2, K1)), (2,1))
	# Alpha =  ones(Float64, (K1, K2)) .* alpha_prior_
	B1 = ones(Float64, (K1,corpus1_.V)) .* rand(Uniform(beta1_prior_/4, beta1_prior_*2), K1)
	B2 = ones(Float64, (K2,corpus2_.V)) .* rand(Uniform(beta2_prior_/4, beta2_prior_*2), K2)
	#variational params
	γ = [ones(Float64, (K1, K2)) for i in 1:N]
	b1 = deepcopy(B1)
	b2 = deepcopy(B2)
	Elog_B1 = zeros(Float64, (K1, corpus1_.V))
	Elog_B2 = zeros(Float64, (K2, corpus2_.V))
	Elog_Theta = [zeros(Float64, (K1, K2)) for i in 1:N]
	## Also make sure vocab is the ones used.
	return 	Alpha,B1,B2,Elog_B1,Elog_B2,Elog_Theta,γ,b1,b2
end

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
function update_Elogtheta!(Elog_, γ_)
	Elog_ .= Elog.(γ_)
end
# function update_Elogtheta!(γ_, Elog_, mb_)
# 	for i in mb_
# 		digsum = digamma_(sum(γ_[i]))
# 		Elog_[i,:,:] .=  digamma_.(γ_[i]) .- digsum
# 	end
# end

function update_Elogtheta_i!(Elog_::Matrix{Float64},γ_::Matrix{Float64})
	Elog_ .= Elog(γ_)
end
function find_all(val, doc)
	findall(x -> x == val, doc)
end
function fix_corp!(model)

	c1 = deepcopy(model.Corpus1)
	for i in 1:length(model.Corpus1.docs)
		doc1 = model.Corpus1.docs[i]
		uniqs1 = unique(doc1.terms)
		counts1 = Int64[]
		for u in uniqs1
			counts1 = vcat(counts1, length(find_all(u, doc1.terms)))
		end
		c1.docs[i] = Document(uniqs1,counts1,doc1.len)

	end
	c2 = deepcopy(model.Corpus2)
	for i in 1:length(model.Corpus2.docs)
		doc2 = model.Corpus2.docs[i]
		uniqs2 = unique(doc2.terms)
		counts2 = Int64[]
		for u in uniqs2
			counts2 = vcat(counts2, length(find_all(u, doc2.terms)))
		end
		c2.docs[i] = Document(uniqs2,counts2,doc2.len)
	end
	model.Corpus1 = c1
	model.Corpus2 = c2
end
function update_Elogb!(Elog_,b_)
	for k in 1:size(Elog_,1)
		Elog_[k,:] .=  Elog(b_[k,:])
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

function optimize_γi!(K1_::Int64, K2_::Int64, Alpha_::Matrix{Float64},γ_::Matrix{Float64}, sum_phi1_::Matrix{Float64}, sum_phi2_::Matrix{Float64})
	@.(γ_ = Alpha_ + sum_phi1_ + sum_phi2_)

end
function optimize_γi!(model::MVD, i,sum_phi1_::Matrix{Float64}, sum_phi2_::Matrix{Float64})
	@.(model.γ[i] = model.Alpha + sum_phi1_ + sum_phi2_)
end
function optimize_b(len_mb, B1, sum_phi_mb, V::Int64, K::Int64, N::Int64)
	b_ = deepcopy(B1)
	b_ .+= (N/len_mb) .* sum_phi_mb
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

function get_lr(epoch, S, κ)
	###Should this be epoch based or iter based?
	return (S+epoch)^(-κ)
end


##### w , and v needs to be fixed in holdout ho and obs what to save probably their indices that correposnd
function calc_theta_bar_i(obs1, obs2,Corpus1, Corpus2, i, model, count_params, w_in_phi_1, w_in_phi_2)

	update_Elogtheta_i!(model.Elog_Theta[i],model.γ[i])
	doc1 = deepcopy(obs1[i])
	doc2 = deepcopy(obs2[i])
	corp1 = deepcopy(Corpus1.docs[i].terms)
	corp2 = deepcopy(Corpus2.docs[i].terms)

	sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
	sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
	#############
	for _u in 1:30
		sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
		for val in unique(corp1[w_in_phi_1[i]])
			x = findall(x -> x == val, corp1[w_in_phi_1[i]])

			y = optimize_phi_iw(model, i,1,val)

			sum_phi_1_i .+= length(x).*y
		end
		sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
		for val in unique(corp2[w_in_phi_2[i]])
			x = findall(x -> x == val, corp2[w_in_phi_2[i]])
			y = optimize_phi_iw(model, i, 2, val)
			# for xx in x
			# 	phi2[i][xx,:, :] .= y
			# end
			sum_phi_2_i .+= length(x).*y
		end
		optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
		update_Elogtheta_i!(model.Elog_Theta[i],model.γ[i])
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
	theta_bar = model.γ[i][:,:] ./ sum(model.γ[i])
	return theta_bar
end


function calc_perp(obs1, obs2,ho1, ho2,Corpus1,Corpus2,model, count_params, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2, B1_est, B2_est)
	corp1 = deepcopy(Corpus1)
	corp2 = deepcopy(Corpus2)
	l1 = 0.0
	l2 = 0.0
	for i in collect(keys(ho1))

		theta_bar = calc_theta_bar_i(obs1, obs2,corp1, corp2, i, model, count_params, w_in_phi_1, w_in_phi_2)


		for w in w_in_ho_1[i]
			v = corp1.docs[i].terms[w]
			tmp = 0.0
			for k in 1:count_params.K1
				tmp += ((B1_est[k,v]*sum(theta_bar, dims=2)[k,1]))
			end
			l1 += log(tmp)
		end
		for w in w_in_ho_2[i]
			v = corp2.docs[i].terms[w]
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
