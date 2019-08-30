include("funcs.jl")

using DataFrames
using DelimitedFiles
using ArgParse
using FileIO
using JLD2

function main(args)
	s = ArgParseSettings()
    @add_arg_table s begin
        # "--file"            #File to read the network from
        #     arg_type = String
        #     required = true
		"-n"               #number of communities
            help = "number of docs"
            arg_type=Int64
            default=2000
		"--k1"               #number of communities
            help = "number of topics in mode 1"
            arg_type=Int64
            default=5
		"--k2"               #number of communities
            help = "number of topics in mode 2"
            arg_type=Int64
            default=5
		"--v1"               #number of communities
            help = "number of vocabs in mode 1"
            arg_type=Int64
            default=200
		"--v2"               #number of communities
            help = "number of vocabs in mode 2"
            arg_type=Int64
            default=200
		"--wlen1"               #number of communities
            help = "number of words per doc 1"
            arg_type=Int64
            default=200
		"--wlen2"               #number of communities
            help = "number of words per doc 2"
            arg_type=Int64
            default=200
        "--mbsize"
            help = "number of docs in a minibatch"
            arg_type=Int64
            default=64
        "--maxiter"
            help = "maximum number of iterations"
            arg_type=Int64
            default=10000
        "--every"
            help = "eval every number of iterations"
            arg_type=Int64
            default=200
		"--kappa"
			help = "kappa for learning rate"
			arg_type = Float64
			default = .5
		"--beta1"
			help = "beta1 prior"
			arg_type = Float64
			default = .3
		"--beta2"
			help = "beta2 prior"
			arg_type = Float64
			default = .3
		"-S"
			help = "S for learning rate"
			arg_type = Float64
			default = 4096.0
		"--holdout"
			help = "holdout"
			arg_type = Float64
			default = .1
    end

    # # #
    parsed_args = ArgParse.parse_args(args,s) ##result is a Dict{String, Any}
    println("Parsed args: ")
    for (k,v) in parsed_args
        println("  $k  =>  $(repr(v))")
    end
    println("before parsing")

    global N = parsed_args["n"]
	global K1 = parsed_args["k1"]
	global K2 = parsed_args["k2"]
	global V1 = parsed_args["v1"]
	global V2 = parsed_args["v2"]
	global β1_single_truth = parsed_args["beta1"]
	global β2_single_truth = parsed_args["beta2"]
	global wlen1_single = parsed_args["wlen1"]
	global wlen2_single = parsed_args["wlen2"]
	global S = parsed_args["S"]
	global κ = parsed_args["kappa"]
	global every = parsed_args["every"]
	global MAXITER = parsed_args["maxiter"]
	global mb_size = parsed_args["mbsize"]
	global h = parsed_args["holdout"]





	# global N = 2000
	# global K1 = 5
	# global K2 = 5
	# global V1 = 200
	# global V2 = 200
	# global β1_single_truth = .3
	# global β2_single_truth = .3
	# global wlen1_single = 200
	# global wlen2_single = 200
	# global S = 256
	# global κ = .5
	# global every = 10
	# global MAXITER = 5000
	# global mb_size = 64
	# global h = 0.01

	global folder = mkdir("n_$(N)_k1k2_$(K1)_$(K2)_v1v2_$(V1)_$(V2)_mb_$(mb_size)_S_$(S)_h_$(h)_max_$(MAXITER)_k_$(κ)")
#########################
	# N, K1, K2, V1, V2 = 2000, 5, 5, 200, 200
	# β1_single_truth, β2_single_truth = .3, .3
	# wlen1_single, wlen2_single = 200, 200
	α,Α, θ,Θ, Β1, Β2, β1, β2, V1, V2, corp1, corp2 =
	 Create_Truth(N, K1, K2, V1, V2, β1_single_truth, β2_single_truth, wlen1_single, wlen2_single)

	 α_truth,Α_truth, θ_truth,Θ_truth,Β1_truth, Β2_truth, β1_truth, β2_truth,V1, V2, corp1, corp2=
	 simulate_data(N, K1, K2, V1, V2,β1_single_truth, β2_single_truth,wlen1_single, wlen2_single)


	Truth_Params = Params(N,K1,K2,V1,V2,α_truth,Α_truth,θ_truth,Θ_truth,β1_truth,β2_truth,Β1_truth,Β2_truth)
	@save "$(folder)/truth" Truth_Params
	Corpus1 = Corpus(N, V1, length.(corp1), corp1)
	@save "$(folder)/corpus1" Corpus1
	Corpus2 = Corpus(N, V2, length.(corp2), corp2)
	@save "$(folder)/corpus2" Corpus2
#########################

	ctm_1 = ctm(Corpus1)
	dtm_1 = dtm(Corpus1)
	ctm_2 = ctm(Corpus2)
	dtm_2 = dtm(Corpus2)

	h_map = setup_train_test(h, Corpus1.N, Corpus1, Corpus2)

	@save "$(folder)/h_map" h_map

	epoch_count=0
	# dtm_train_1 = dtm(train_corp_1)
	# dtm_train_2 = dtm(train_corp_2)


	alpha_vec, Alpha,beta1, beta2,phi1, phi2, γ, b1, b2, Elog_B1, Elog_B2, Elog_Theta =
	init_params(Truth_Params, .5, .5, .5, Corpus1, Corpus1);

	update_Elogtheta!(γ, Elog_Theta)

	update_Elogb!(b1, Elog_B1)
	update_Elogb!(b2, Elog_B2)
	#γ_old = deepcopy(γ)

	# mb_size = 64
	mbs, nb = epoch_batches(N, mb_size, h_map)

	mindex = 1
	count_params = CountParams(Corpus1.N-sum(h_map), K1, K2)
	println("VI Started")
	ho1, obs1, ho2, obs2, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2 =
	create_test(h_map, Corpus1, Corpus2)
	l1_list = Float64[]
	l2_list = Float64[]
	for iter in 1:MAXITER

		# global mindex, nb, mbs, count_params,mb_size, l1_list, l2_list,epoch_count
		if mindex == (nb+1)
			mbs, nb = epoch_batches(N, mb_size, h_map)
			mindex = 1
			theta_est = estimate_thetas(γ)
			B1_est = estimate_B(b1)
			B2_est = estimate_B(b2)
			if epoch_count % every == 0
				println("starting to calc perp")
				l1, l2 = calc_perp(obs1, obs2,ho1,ho2,Corpus1, Corpus2, γ, Alpha, Elog_Theta,
				 Elog_B1,Elog_B2, count_params, phi1, phi2, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2, B1_est, B2_est)
				 l1_list = vcat(l1_list, l1)
				 println("l1=$(l1)")
				 l2_list = vcat(l2_list, l2)
				 println("l2=$(l2)")
				@save "$(folder)/theta_at_epoch_$(epoch_count)"  theta_est
				@save "$(folder)/B1_at_epoch_$(epoch_count)"  B1_est
				@save "$(folder)/B2_at_epoch_$(epoch_count)"  B2_est
			end

		end
		if mindex  == nb
			epoch_count +=1
			if epoch_count % every == 0
				println("i:$(iter) epoch :$(epoch_count)")

			end

		end
		mb = mbs[mindex]

		# ρ = get_lr(epoch_count, S, κ)
		ρ = get_lr(iter, S, κ)

		for i in mb

			#γ[i] =  ones(Float64, (K1, K2))
			Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	 		doc1 = deepcopy(Corpus1.Data[i])
	 		doc2 = deepcopy(Corpus2.Data[i])
			for _u in 1:3
				if rand() > .5
			 		for (w, val) in enumerate(doc1)
			 			phi1[i][w,:, :] .= optimize_phi1_iw(phi1[i], Elog_Theta[i,:,:],Elog_B1, count_params, w, doc1)
			 		end

			 		for (w,val) in enumerate(doc2)
			 			phi2[i][w,:, :] .= optimize_phi2_iw(phi2[i], Elog_Theta[i,:,:],Elog_B2, count_params, w, doc2)
			 		end
				else
					for (w,val) in enumerate(doc2)
						phi2[i][w,:, :] .= optimize_phi2_iw(phi2[i], Elog_Theta[i,:,:],Elog_B2, count_params, w, doc2)
					end
					for (w, val) in enumerate(doc1)
			 			phi1[i][w,:, :] .= optimize_phi1_iw(phi1[i], Elog_Theta[i,:,:],Elog_B1, count_params, w, doc1)
			 		end
				end
				optimize_γi!(count_params.K1, count_params.K2, Alpha,γ[i], phi1[i], phi2[i])
				Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
			end
				#γ_old[i] .= deepcopy(γ[i])

			# if gamma_converged(γ[i], γ_old[i])
			# 	println("$i converged")
			# 	break;
			# end
		end

		old_b1 = deepcopy(b1)
		b1 .= optimize_b1(mb, beta1, phi1, Corpus1, count_params)
		b1 .= (1-ρ).*old_b1 + ρ.*b1
		update_Elogb!(b1, Elog_B1)
		old_b2 = deepcopy(b2)
 		b2 .= optimize_b2(mb, beta2, phi2, Corpus2, count_params)
		b2 .= (1-ρ).*old_b2 + ρ.*b2
		update_Elogb!(b2, Elog_B2)




		mindex += 1
		if iter == MAXITER

			println("Final rounds")
			mb = collect(1:N)
			for _ in 1:5
				for i in mb

					# γ[i] =  ones(Float64, (K1, K2))
					Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
					doc1 = deepcopy(Corpus1.Data[i])
			 		doc2 = deepcopy(Corpus2.Data[i])
					for _u in 1:3
						if rand() > .5
					 		for (w, val) in enumerate(doc1)
					 			phi1[i][w,:, :] .= optimize_phi1_iw(phi1[i], Elog_Theta[i,:,:],Elog_B1, count_params, w, doc1)
					 		end

					 		for (w,val) in enumerate(doc2)
					 			phi2[i][w,:, :] .= optimize_phi2_iw(phi2[i], Elog_Theta[i,:,:],Elog_B2, count_params, w, doc2)
					 		end
						else
							for (w,val) in enumerate(doc2)
								phi2[i][w,:, :] .= optimize_phi2_iw(phi2[i], Elog_Theta[i,:,:],Elog_B2, count_params, w, doc2)
							end
							for (w, val) in enumerate(doc1)
					 			phi1[i][w,:, :] .= optimize_phi1_iw(phi1[i], Elog_Theta[i,:,:],Elog_B1, count_params, w, doc1)
					 		end
						end
						optimize_γi!(count_params.K1, count_params.K2, Alpha,γ[i], phi1[i], phi2[i])
						Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
					end
						#γ_old[i] .= deepcopy(γ[i])
				end

				b1 .= optimize_b1(mb, beta1, phi1, Corpus1, count_params)
				update_Elogb!(b1, Elog_B1)
		 		b2 .= optimize_b2(mb, beta2, phi2, Corpus2, count_params)
				update_Elogb!(b2, Elog_B2)
			end
		end
	end
	theta_est = estimate_thetas(γ)
	@save "$(folder)/theta_last"  theta_est
	B1_est = estimate_B(b1)
	@save "$(folder)/B1_last"  B1_est
	B2_est = estimate_B(b2)
	@save "$(folder)/B2_last"  B2_est

	@save "$(folder)/l1_list"  l1_list
	@save "$(folder)/l2_list"  l2_list

end

main(ARGS)
# using Plots
#
# Plots.heatmap(B1_est, yflip = true)
# Plots.heatmap(Β1_truth, yflip = true)
# Plots.heatmap(B1_est[[2,5,3,1,4],:], yflip = true)
# idx1 = [2,5,3,1,4]
# ###########
# Plots.heatmap(B2_est, yflip = true)
# Plots.heatmap(Β2_truth, yflip = true)
# Plots.heatmap(B2_est[[1,3,5,4,2],:], yflip = true)
# idx2 = [1,3,5,4,2]
#
#
# Plots.heatmap(theta_est[12][idx1, idx2 ], yflip = true)
# Plots.heatmap(Θ_truth[12], yflip = true)
