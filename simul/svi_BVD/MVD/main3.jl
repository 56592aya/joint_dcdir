include("loader3.jl")



function main(args)
	s = ArgParseSettings()
    @add_arg_table s begin
		"--data"            #data folder
            arg_type = String
            required = true
		"--all"               #if sparsity for all
            help = "If sparsity for all"
            action = "store_true"
		"--sparsity"               #sparsity
            help = "percent not available"
            arg_type=Float64
            default=.5
		"--k1"               #number of communities
            help = "number of topics in mode 1"
            arg_type=Int64
            default=5
		"--k2"               #number of communities
            help = "number of topics in mode 2"
            arg_type=Int64
            default=5
        "--mbsize"
            help = "number of docs in a minibatch"
            arg_type=Int64
            default=64
        "--maxiter"
            help = "maximum number of iterations"
            arg_type=Int64
            default=5000
        "--every"
            help = "eval every number of iterations"
            arg_type=Int64
            default=10
		"--kappa"
			help = "kappa for learning rate"
			arg_type = Float64
			default = .5
		"--alpha_prior"
			help = "alpha prior"
			arg_type = Float64
			default = .3
		"--beta1_prior"
			help = "beta1 prior"
			arg_type = Float64
			default = .3
		"--beta2_prior"
			help = "beta2 prior"
			arg_type = Float64
			default = .3
		"-S"
			help = "S for learning rate"
			arg_type = Float64
			default = 256.0
		"--holdout"
			help = "holdout"
			arg_type = Float64
			default = .01
    end
    # # #
    parsed_args = ArgParse.parse_args(args,s) ##result is a Dict{String, Any}
    @info "Parsed args: "
    for (k,v) in parsed_args
        @info "  $k  =>  $(repr(v))"
    end
    @info "before parsing"

	global data_folder = parsed_args["data"]
	global K1 = parsed_args["k1"]
	global K2 = parsed_args["k2"]
	global α_single_prior = parsed_args["alpha_prior"]
	global β1_single_prior = parsed_args["beta1_prior"]
	global β2_single_prior = parsed_args["beta2_prior"]
	global S = parsed_args["S"]
	global κ = parsed_args["kappa"]
	global every = parsed_args["every"]
	global MAXITER = parsed_args["maxiter"]
	global mb_size = parsed_args["mbsize"]
	global h = parsed_args["holdout"]
	global all_ = parsed_args["all"]
	global sparsity = parsed_args["sparsity"]



	 # global K1 = 5
	 # global K2 = 5
 	 # global α_single_prior = .1
 	 # global β1_single_prior = .1
	 # global β2_single_prior = .1
	 # global S = 64.0
	 # global κ = .7
	 # global every = 5
	 # global MAXITER = 100000
	 # global mb_size = 256
	 # global h = 0.005
	 # global data_folder = "3000_5_5_100_100_0.1_0.1_0.1_0.0"
	 # global all_ = true
 	 # global sparsity = 0.0

	global folder = mkdir(joinpath(data_folder,"est_$(K1)_$(K2)_$(mb_size)_$(MAXITER)_$(h)_$(S)_$(κ)_$(every)_$(α_single_prior)_$(β1_single_prior)_$(β2_single_prior)_$(all_)_$(sparsity)"))

#########################


	@load "$(data_folder)/corpus1" Corpus1
	@load "$(data_folder)/corpus2" Corpus2
	# using JLD2, FileIO
	# @load "../5000_5_10_200_400_0.2_0.1_0.1/corpus1" Corpus1
	# @load "../5000_5_10_200_400_0.2_0.1_0.1/corpus2" Corpus2

	global N = max(Corpus1.N, Corpus2.N)
	init_params_ = init_params(K1, K2, β1_single_prior, β2_single_prior, α_single_prior, Corpus1, Corpus2);
	model = MVD(K1, K2, Corpus1, Corpus2, init_params_...)







	### BEGIN Fix Corpuses and figure sparsity in Corpus2
	fix_corp!(model)
	corp1 = deepcopy(model.Corpus1)
	corp2 = deepcopy(model.Corpus2)
	@save "$(folder)/Corpus1" corp1
	@save "$(folder)/Corpus2" corp2
	if h != 0.0
		figure_sparsity!(model,sparsity,all_)
		corp2_sparse = deepcopy(model.Corpus2)
		@save "$(folder)/Corpus2_sparse" corp2_sparse
	else
		
	end
	### END Fix Corpuses and figure sparsity in Corpus2

	### BEGIN setting up train, holdout and test


	h_map = setup_hmap(model, h)
	@save "$(folder)/h_map" h_map

	mbs, nb = epoch_batches(N, mb_size, h_map)
	mindex, epoch_count = 1,0
	hos1_dict,obs1_dict,hos2_dict,obs2_dict =split_ho_obs(model, h_map)
	count_params = CountParams(model.Corpus1.N-sum(h_map), model.K1, model.K2)
	### END setting up train, holdout and test
	update_Elogtheta!(model.Elog_Theta, model.γ)
	update_Elogb!(model.Elog_B1,model.b1)
	update_Elogb!(model.Elog_B2, model.b2)


	perp1_list = Float64[]
	perp2_list = Float64[]
	@info "VI Started"
	global converged = false
	for iter in 1:MAXITER


		# global model, mindex, nb, mbs, count_params,mb_size, perp1_list, perp2_list,epoch_count
		if mindex == (nb+1)

			mbs, nb = epoch_batches(N, mb_size, h_map)
			mindex = 1
			theta_est = estimate_thetas(model.γ)
			B1_est = estimate_B(model.b1)
			B2_est = estimate_B(model.b2)
			Alpha_est = deepcopy(model.Alpha)
			if epoch_count % every == 0
				@info "starting to calc perp"
				p1, p2 = calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
				 count_params, B1_est, B2_est)

				 perp1_list = vcat(perp1_list, p1)
				 @info "perp1=$(p1)"
				 perp2_list = vcat(perp2_list, p2)
				 @info "perp2=$(p2)"
				@save "$(folder)/theta_at_epoch_$(epoch_count)"  theta_est
				@save "$(folder)/B1_at_epoch_$(epoch_count)"  B1_est
				@save "$(folder)/B2_at_epoch_$(epoch_count)"  B2_est
				@save "$(folder)/Alpha_at_epoch_$(epoch_count)"  Alpha_est
				@save "$(folder)/perp1_at_$(epoch_count)"  perp1_list
				@save "$(folder)/perp2_at_$(epoch_count)"  perp2_list

				if length(perp1_list) > 2
					if (abs(perp1_list[end]-perp1_list[end-1])/perp1_list[end] < 1e-8) &&
						(abs(perp2_list[end]-perp2_list[end-1])/perp2_list[end] < 1e-8)
						converged  = true
					end
				end
			end
		end

		if mindex  == nb
			epoch_count +=1
			if epoch_count % every == 0
				@info "i:$(iter) epoch :$(epoch_count)"

			end
		end

		mb = mbs[mindex]

		ρ = get_lr(iter, S, κ)
		sum_phi_1_mb = zeros(Float64, (count_params.K1,model.Corpus1.V))
		sum_phi_2_mb = zeros(Float64, (count_params.K2,model.Corpus2.V))
		zeroer_i = zeros(Float64, (count_params.K1, count_params.K2))
		sum_phi_1_i = deepcopy(zeroer_i)
		sum_phi_2_i = deepcopy(zeroer_i)
		for i in mb

			update_Elogtheta_i!(model, i)

	 		doc1 = deepcopy(model.Corpus1.docs[i])
	 		doc2 = deepcopy(model.Corpus2.docs[i])
			γ_old = deepcopy(model.γ[i])
			gamma_c = false

			for _u in 1:30

				# global model,gamma_c,γ_old, sum_phi_1_mb, sum_phi_2_mb
				copyto!(sum_phi_1_i, zeroer_i)
				for (w,val) in enumerate(doc1.terms)

					y = optimize_phi_iw(model, i,1,val)

					sum_phi_1_i .+= doc1.counts[w] .* y
					if (_u == 30) || gamma_c
						sum_phi_1_mb[:,val] .+= sum(doc1.counts[w].* y, dims = 2)[:,1]
					end
				end

				copyto!(sum_phi_2_i, zeroer_i)
				for (w,val) in enumerate(doc2.terms)
					y = optimize_phi_iw(model, i,2,val)
					sum_phi_2_i .+= doc2.counts[w].*y
					if (_u == 30) || gamma_c
						sum_phi_2_mb[:,val] .+= sum(doc2.counts[w] .* y , dims = 1)[1,:]
					end
				end
				optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
				update_Elogtheta_i!(model,i)

				if (mean(abs.(γ_old .- model.γ[i])./model.γ[i])) < 1e-4
					gamma_c = true
				end
				γ_old = deepcopy(model.γ[i])
	 		end

		end # i in mb end
		old_b1 = deepcopy(model.b1)
		model.b1 .= optimize_b(length(mb), model.B1, sum_phi_1_mb, count_params)
		model.b1 .= (1-ρ).*old_b1 + ρ.*model.b1
		update_Elogb!(model.Elog_B1, model.b1)
		old_b2 = deepcopy(model.b2)
		model.b2 .= optimize_b(length(mb), model.B2, sum_phi_2_mb,count_params)
		model.b2 .= (1-ρ).*old_b2 + ρ.*model.b2
		update_Elogb!(model.Elog_B2, model.b2)

		if epoch_count % every == 0
			# ρ_alpha = get_lr(div(epoch_count,every)+1, S, κ)
			old_alpha = deepcopy(model.Alpha)
			update_alpha!(model, mb, .5)
			model.Alpha .= (1-ρ).*old_alpha + ρ.*model.Alpha
		end

		mindex += 1
							  ################################
									###For FINAL Rounds###
							  ################################
		if iter == MAXITER || converged
			@info "Final rounds"
			mb = collect(1:N)[.!h_map]
			sum_phi_1_mb = zeros(Float64, (count_params.K1,model.Corpus1.V))
			sum_phi_2_mb = zeros(Float64, (count_params.K2,model.Corpus2.V))
			zeroer_i = zeros(Float64, (count_params.K1, count_params.K2))
			sum_phi_1_i = deepcopy(zeroer_i)
			sum_phi_2_i = deepcopy(zeroer_i)
			for i in mb

				# @btime Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
				# Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
				update_Elogtheta_i!(model, i)
		 		doc1 = deepcopy(model.Corpus1.docs[i])
		 		doc2 = deepcopy(model.Corpus2.docs[i])
				γ_old = deepcopy(model.γ[i])
				gamma_c = false

				for _u in 1:30
					# global model,gamma_c,γ_old, sum_phi_1_mb, sum_phi_2_mb
					copyto!(sum_phi_1_i, zeroer_i)
					for (w,val) in enumerate(doc1.terms)
						# x = find_all(val, doc1.terms)
						# y = optimize_phi1_iw_2(Elog_Theta[i,:,:],Elog_B1, count_params,val)
						y = optimize_phi_iw(model, i,1,val)
						# for xx in x
						# 	copyto!(phi1[i][xx,:, :],y)
						# end
						sum_phi_1_i .+= doc1.counts[w].*y

						if (_u == 30) || gamma_c
							sum_phi_1_mb[:,val] .+= sum(doc1.counts[w].* y, dims = 2)[:,1]
						end
					end

					copyto!(sum_phi_2_i, zeroer_i)
					for (w,val) in enumerate(doc2.terms)
						# x = find_all(val, doc2.terms)
						# y = optimize_phi2_iw_2(Elog_Theta[i,:,:],Elog_B2, count_params,val)
						y = optimize_phi_iw(model, i,2,val)
						# for xx in x
						# 	copyto!(phi2[i][xx,:, :],y)
						# end
						sum_phi_2_i .+= doc2.counts[w].*y
						if (_u == 30) || gamma_c
							sum_phi_2_mb[:,val] .+= sum(doc2.counts[w].* y, dims = 1)[1,:]
						end
					end
					optimize_γi!(model, i, sum_phi_1_i, sum_phi_2_i)
					# Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
					update_Elogtheta_i!(model, i)


					# println(mean(abs.(γ_old .- γ[i])/γ[i]))
					if (mean(abs.(γ_old .- model.γ[i])/model.γ[i])) < 1e-4
						gamma_c = true
					end
					γ_old = deepcopy(model.γ[i])
		 		end

			end # i in mb end

			model.b1 .= optimize_b(length(mb), model.B1, sum_phi_1_mb, Corpus1.V, count_params.K1, count_params.N)
			update_Elogb!(model.Elog_B1, model.b1)
			model.b2 .= optimize_b(length(mb), model.B2, sum_phi_2_mb, Corpus2.V, count_params.K2, count_params.N)
			update_Elogb!(model.Elog_B2, model.b2)
			update_alpha!(model, collect(1:count_params.N), .5)
			break
		end

	end
	theta_est = estimate_thetas(model.γ)
	@save "$(folder)/theta_last"  theta_est
	B1_est = estimate_B(model.b1)
	@save "$(folder)/B1_last"  B1_est
	B2_est = estimate_B(model.b2)
	@save "$(folder)/B2_last"  B2_est
	Alpha_est = deepcopy(model.Alpha)
	@save "$(folder)/Alpha_last"  Alpha_est

	@save "$(folder)/perp1_list"  perp1_list
	@save "$(folder)/perp2_list"  perp2_list

end

main(ARGS)
