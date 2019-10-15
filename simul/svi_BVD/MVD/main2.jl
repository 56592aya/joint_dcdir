include("loader.jl")



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



	 # global K1 = 4
	 # global K2 = 4
 	 # global α_single_prior = .98
 	 # global β1_single_prior = .98
	 # global β2_single_prior = .98
	 # global S = 1024.0
	 # global κ = .6
	 # global every = 5
	 # global MAXITER = 100000
	 # global mb_size = 128
	 # global h = 0.01
	 # global data_folder = "5000_4_4_100_100_1.0_0.25_0.25_0.0"
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
	figure_sparsity!(model,sparsity,all_)
	corp2_sparse = deepcopy(model.Corpus2)
	@save "$(folder)/Corpus2_sparse" corp2_sparse
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
	update_Elogb!(model,1)
	update_Elogb!(model,2)


	perp1_list = Float64[]
	perp2_list = Float64[]
	@info "VI Started"
	global converged = false


	global zeroer_i = zeros(Float64, (count_params.K1, count_params.K2))
	global zeroer_mb_1 = zeros(Float64, (count_params.K1,model.Corpus1.V))
	global zeroer_mb_2 = zeros(Float64, (count_params.K2,model.Corpus2.V))

	for iter in 1:MAXITER
		# iter = 1
		# global model, mindex, nb, mbs, count_params,mb_size, perp1_list, perp2_list,epoch_count

		if mindex == (nb+1) || iter == 1

			mbs, nb = epoch_batches(N, mb_size, h_map)
			mindex = 1
			# theta_est = estimate_thetas(model.γ)
			B1_est = estimate_B(model.b1)
			B2_est = estimate_B(model.b2)
			# Alpha_est = deepcopy(model.Alpha)
			if (epoch_count % every == 0) || (epoch_count == 0)
				@info "starting to calc perp"
				p1, p2 = calc_perp(model,hos1_dict,obs1_dict,hos2_dict,obs2_dict,
				 count_params, B1_est, B2_est,zeroer_i)

				 perp1_list = vcat(perp1_list, p1)
				 @info "perp1=$(p1)"
				 perp2_list = vcat(perp2_list, p2)
				 @info "perp2=$(p2)"
				@save "$(folder)/model_at_epoch_$(epoch_count)"  model
				# @save "$(folder)/B1_at_epoch_$(epoch_count)"  B1_est
				# @save "$(folder)/B2_at_epoch_$(epoch_count)"  B2_est
				# @save "$(folder)/Alpha_at_epoch_$(epoch_count)"  Alpha_est
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
		copy!(model.sum_phi_1_mb, zeroer_mb_1)
		copyto!(model.sum_phi_2_mb, zeroer_mb_2)
		copyto!(model.sum_phi_1_i,  zeroer_i)
		copyto!(model.sum_phi_2_i, zeroer_i)

		for i in mb
			update_Elogtheta_i!(model, i)
	 		doc1 = model.Corpus1.docs[i]
	 		doc2 = model.Corpus2.docs[i]
			copyto!(model.old_γ, model.γ[i])
			gamma_c = false
			update_phis_gammas!(model, i,zeroer_i,doc1,doc2,gamma_c)
		end # i in mb end
		copyto!(model.old_b1,  model.b1)
		optimize_b!(model.b1, length(mb), model.B1, model.sum_phi_1_mb, count_params)
		model.b1 .= (1.0-ρ).*model.old_b1 .+ ρ.*model.b1
		update_Elogb!(model, 1)
		copyto!(model.old_b2,model.b2)
		optimize_b!(model.b2,length(mb), model.B2, model.sum_phi_2_mb,count_params)
		model.b2 .= (1.0-ρ).*model.old_b2 .+ ρ.*model.b2
		update_Elogb!(model, 2)

		if (mindex == nb)
			copyto!(model.old_Alpha,model.Alpha)
			update_alpha!(model, mb,1.0, count_params)
			model.Alpha .= (1.0-ρ).*model.old_Alpha .+ ρ.*model.Alpha
			copyto!(model.old_B1, model.B1)
			for k in 1:model.K1
				update_beta1!(model,k, 1.0)
				model.B1[k,:] .= (1.0-ρ).*model.old_B1[k,:] .+ ρ.*model.B1[k,:]
			end

			copyto!(model.old_B2, model.B2)
			for k in 1:model.K2
				update_beta2!(model,k, 1.0)
				model.B2[k,:] .= (1.0-ρ).*model.old_B2[k,:] .+ ρ.*model.B2[k,:]
			end
		end

		mindex += 1
		# iter +=1

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
				update_Elogtheta_i!(model, i)
		 		doc1 = deepcopy(model.Corpus1.docs[i])
		 		doc2 = deepcopy(model.Corpus2.docs[i])
				γ_old = deepcopy(model.γ[i])
				gamma_c = false
				update_phis_gammas!(model, i, sum_phi_1_i, sum_phi_2_i,sum_phi_1_mb,sum_phi_2_mb,zeroer_i,doc1,doc2,gamma_c,γ_old)
			end # i in mb end
			optimize_b!(model.b1, length(mb), model.B1, model.sum_phi_1_mb, count_params)
			update_Elogb!(model, 1)
			optimize_b!(model.b2,length(mb), model.B2, model.sum_phi_2_mb,count_params)
			update_Elogb!(model, 2)
			# update_alpha!(model, collect(1:count_params.N), .5)
			break
		end

	end
	theta_est = estimate_thetas(model.γ)
	# @save "$(folder)/theta_last"  theta_est
	B1_est = estimate_B(model.b1)
	# @save "$(folder)/B1_last"  B1_est
	B2_est = estimate_B(model.b2)
	# @save "$(folder)/B2_last"  B2_est
	Alpha_est = deepcopy(model.Alpha)
	# @save "$(folder)/Alpha_last"  Alpha_est
	@save "$(folder)/model_at_last"  model

	@save "$(folder)/perp1_list"  perp1_list
	@save "$(folder)/perp2_list"  perp2_list

end

main(ARGS)
