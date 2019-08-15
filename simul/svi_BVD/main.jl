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
			arg_type = Int64
			default = 4096
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

	global folder = mkdir("n_$(N)_k1k2_$(K1)_$(K2)_v1v2_$(V1)_$(V2)_mb_$(mb_size)")
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
	select_from_1 = valid_holdout_selection(dtm_1, 3)
	select_from_2 = valid_holdout_selection(dtm_2, 3)
	(train_corp_1, holdout_1) = train_holdout(h,Corpus1, select_from_1, dtm_1)
	@save "$(folder)/train_1" train_corp_1
	@save "$(folder)/holdout_1" holdout_1
	(train_corp_2, holdout_2) = train_holdout(h,Corpus2, select_from_2, dtm_2)
	@save "$(folder)/train_2" train_corp_2
	@save "$(folder)/holdout_2" holdout_2
	epoch_count=1
	dtm_train_1 = dtm(train_corp_1)
	dtm_train_2 = dtm(train_corp_2)


	alpha_vec, Alpha,beta1, beta2,phi1, phi2, γ, b1, b2, Elog_B1, Elog_B2, Elog_Theta =
	init_params(Truth_Params, .5, .5, .5, train_corp_1, train_corp_2);

	update_Elogtheta!(γ, Elog_Theta)

	update_Elogb!(b1, Elog_B1)
	update_Elogb!(b2, Elog_B2)
	γ_old = deepcopy(γ)

	# mb_size = 64
	mbs, nb = epoch_batches(N, mb_size)

	mindex = 1
	count_params = CountParams(train_corp_1.N, K1, K2)
	for iter in 1:MAXITER

		# global mindex, nb, mbs, count_params,mb_size
		if mindex == (nb+1)
			mbs, nb = epoch_batches(N, mb_size)
			mindex = 1
			theta_est = estimate_thetas(γ)
			@save "$(folder)/theta_at_iter_$(iter)"  theta_est
			B1_est = estimate_B(b1)
			@save "$(folder)/B1_at_iter_$(iter)"  B1_est
			B2_est = estimate_B(b2)
			@save "$(folder)/B2_at_iter_$(iter)"  B2_est

		end
		if mindex  == nb
			epoch_count +=1
			println("i:$(iter) epoch :$(epoch_count)")
		end
		mb = mbs[mindex]

		ρ = get_lr(iter, S, κ)

		for i in mb

			γ[i] =  ones(Float64, (K1, K2))
			Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	 		doc1 = deepcopy(train_corp_1.Data[i])
	 		doc2 = deepcopy(train_corp_2.Data[i])
			for _u in 1:5
		 		for (w, val) in enumerate(doc1)
		 			phi1[i][w,:, :] .= optimize_phi1_iw(phi1[i], Elog_Theta[i,:,:],Elog_B1, count_params, w, doc1)
		 		end

		 		for (w,val) in enumerate(doc2)
		 			phi2[i][w,:, :] .= optimize_phi2_iw(phi2[i], Elog_Theta[i,:,:],Elog_B2, count_params, w, doc2)
		 		end
				γ_old[i] .= deepcopy(γ[i])
				optimize_γi!(count_params.K1, count_params.K2, Alpha,γ[i], phi1[i], phi2[i])
				Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])

				# if gamma_converged(γ[i], γ_old[i])
				# 	println("$i converged")
				# 	break;
				# end
		 	end
		end

		old_b1 = deepcopy(b1)
		b1 .= optimize_b1(mb, beta1, phi1, train_corp_1, count_params)
		b1 .= (1-ρ).*old_b1 + ρ.*b1
		update_Elogb!(b1, Elog_B1)
		old_b2 = deepcopy(b2)
 		b2 .= optimize_b2(mb, beta2, phi2, train_corp_2, count_params)
		b2 .= (1-ρ).*old_b2 + ρ.*b2
		update_Elogb!(b2, Elog_B2)




		mindex += 1
	end
	if iter == MAXITER
		theta_est = estimate_thetas(γ)
		@save "$(folder)/theta_last"  theta_est
		B1_est = estimate_B(b1)
		@save "$(folder)/B1_last"  B1_est
		B2_est = estimate_B(b2)
		@save "$(folder)/B2_last"  B2_est
	end
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
