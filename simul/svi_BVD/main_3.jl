include("funcs.jl")

using DataFrames
using DelimitedFiles
using ArgParse
using FileIO
using JLD2
using BenchmarkTools
using Logging

function main(args)
	s = ArgParseSettings()
    @add_arg_table s begin
		"--data"            #data folder
            arg_type = String
            required = true
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



	 # global K1 = 5
	 # global K2 = 10
 	 # global α_single_prior = .3
 	 # global β1_single_prior = .3
	 # global β2_single_prior = .3
	 # global S = 256.0
	 # global κ = .7
	 # global every = 5
	 # global MAXITER = 500000
	 # global mb_size = 128
	 # global h = 0.005
	 # data_folder = "5000_5_10_200_400_0.2_0.1_0.3"

	global folder = mkdir("$(data_folder)/est_$(K1)_$(K2)_$(mb_size)_$(MAXITER)_$(h)_$(S)_$(κ)_$(every)_$(α_single_prior)_$(β1_single_prior)_$(β2_single_prior)")
#########################


	@load "$(data_folder)/corpus1" Corpus1
	@load "$(data_folder)/corpus2" Corpus2
	global N = max(Corpus1.N, Corpus2.N)

	alpha_vec, Alpha,beta1, beta2,phi1, phi2, γ, b1, b2, Elog_B1, Elog_B2, Elog_Theta =
				init_params(K1, K2, β1_single_prior, β2_single_prior, α_single_prior, Corpus1, Corpus2);

	update_Elogtheta!(γ, Elog_Theta)
	update_Elogb!(b1, Elog_B1)
	update_Elogb!(b2, Elog_B2)
	h_map = setup_train_test(h, N, Corpus1, Corpus2)
	@save "$(folder)/h_map" h_map
	mbs, nb = epoch_batches(N, mb_size, h_map)

	mindex, epoch_count = 1,0
	count_params = CountParams(Corpus1.N-sum(h_map), K1, K2)
	ho1, obs1, ho2, obs2, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2 =create_test(h_map, Corpus1, Corpus2)
	perp1_list = Float64[]
	perp2_list = Float64[]
	@info "VI Started"
	global converged = false
	for iter in 1:MAXITER

		# global mindex, nb, mbs, count_params,mb_size, perp1_list, perp2_list,epoch_count

		if mindex == (nb+1)
			mbs, nb = epoch_batches(N, mb_size, h_map)
			mindex = 1
			theta_est = estimate_thetas(γ)
			B1_est = estimate_B(b1)
			B2_est = estimate_B(b2)
			if epoch_count % every == 0
				@info "starting to calc perp"
				p1, p2 = calc_perp(obs1, obs2,ho1,ho2,Corpus1, Corpus2, γ, Alpha, Elog_Theta,
				 Elog_B1,Elog_B2, count_params, phi1, phi2, w_in_phi_1, w_in_phi_2,w_in_ho_1,w_in_ho_2, B1_est, B2_est)
				 perp1_list = vcat(perp1_list, p1)
				 @info "perp1=$(p1)"
				 perp2_list = vcat(perp2_list, p2)
				 @info "perp2=$(p2)"
				@save "$(folder)/theta_at_epoch_$(epoch_count)"  theta_est
				@save "$(folder)/B1_at_epoch_$(epoch_count)"  B1_est
				@save "$(folder)/B2_at_epoch_$(epoch_count)"  B2_est
				if length(perp1_list) > 2
					if (abs(perp1_list[end]-perp1_list[end-1])/perp1_list[end] < 1e-8) ||
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
		sum_phi_1_mb = zeros(Float64, (count_params.K1,Corpus1.V))
		sum_phi_2_mb = zeros(Float64, (count_params.K2,Corpus2.V))
		sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
		sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
		for i in mb

			Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	 		doc1 = deepcopy(Corpus1.Data[i])
	 		doc2 = deepcopy(Corpus2.Data[i])
			for _u in 1:5
				sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
				for val in unique(doc1)
					x = findall(x -> x == val, doc1)
					y = optimize_phi1_iw_2(Elog_Theta[i,:,:],Elog_B1, count_params,val)
					for xx in x
						phi1[i][xx,:, :] .= y
					end

					sum_phi_1_i .+= length(x).*y
					if _u == 5
						sum_phi_1_mb[:,val] .+= sum(length(x).* y, dims = 2)[:,1]
					end
				end
				sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
				for val in unique(doc2)
					x = findall(x -> x == val, doc2)
					y = optimize_phi2_iw_2(Elog_Theta[i,:,:],Elog_B2, count_params,val)
					for xx in x
						phi2[i][xx,:, :] .= y
					end
					sum_phi_2_i .+= length(x).*y
					if _u == 5
						sum_phi_2_mb[:,val] .+= sum(length(x).* y, dims = 1)[1,:]
					end
				end
				optimize_γi_2!(count_params.K1, count_params.K2, Alpha,γ[i], sum_phi_1_i, sum_phi_2_i)
				Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	 		end
		end # i in mb end


		old_b1 = deepcopy(b1)
		b1 .= optimize_b(length(mb), beta1, sum_phi_1_mb, Corpus1.V, count_params.K1, count_params.N)
		b1 .= (1-ρ).*old_b1 + ρ.*b1
		update_Elogb!(b1, Elog_B1)
		old_b2 = deepcopy(b2)
		b2 .= optimize_b(length(mb), beta2, sum_phi_2_mb, Corpus2.V, count_params.K2, count_params.N)
		b2 .= (1-ρ).*old_b2 + ρ.*b2
		update_Elogb!(b2, Elog_B2)

		mindex += 1
							  ################################
									###For FINAL Rounds###
							  ################################
		if iter == MAXITER || converged
			@info "Final rounds"
			mb = collect(1:N)[.!h_map]
			sum_phi_1_mb = zeros(Float64, (count_params.K1,Corpus1.V))
			sum_phi_2_mb = zeros(Float64, (count_params.K2,Corpus2.V))
			sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
			sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
			for i in mb
				Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
		 		doc1 = deepcopy(Corpus1.Data[i])
		 		doc2 = deepcopy(Corpus2.Data[i])
				for _u in 1:5
					sum_phi_1_i = zeros(Float64, (count_params.K1, count_params.K2))
					for val in unique(doc1)
						x = findall(x -> x == val, doc1)
						y = optimize_phi1_iw_2(Elog_Theta[i,:,:],Elog_B1, count_params,val)
						for xx in x
							phi1[i][xx,:, :] .= y
						end
						sum_phi_1_i .+= length(x).*y
						if _u == 5
							sum_phi_1_mb[:,val] .+= sum(length(x).* y, dims = 2)[:,1]
						end
					end
					sum_phi_2_i = zeros(Float64, (count_params.K1, count_params.K2))
					for val in unique(doc2)
						x = findall(x -> x == val, doc2)
						y = optimize_phi2_iw_2(Elog_Theta[i,:,:],Elog_B2, count_params,val)
						for xx in x
							phi2[i][xx,:, :] .= y
						end
						sum_phi_2_i .+= length(x).*y
						if _u == 5
							sum_phi_2_mb[:,val] .+= sum(length(x).* y, dims = 1)[1,:]
						end
					end
					optimize_γi_2!(count_params.K1, count_params.K2, Alpha,γ[i], sum_phi_1_i, sum_phi_2_i)
					Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
		 		end
			end
			b1 .= optimize_b(length(mb), beta1, sum_phi_1_mb, Corpus1.V, count_params.K1, count_params.N)
			update_Elogb!(b1, Elog_B1)
	 		b2 .= optimize_b(length(mb), beta2, sum_phi_2_mb, Corpus2.V, count_params.K2, count_params.N)
			update_Elogb!(b2, Elog_B2)
			break
		end

	end
	theta_est = estimate_thetas(γ)
	@save "$(folder)/theta_last"  theta_est
	B1_est = estimate_B(b1)
	@save "$(folder)/B1_last"  B1_est
	B2_est = estimate_B(b2)
	@save "$(folder)/B2_last"  B2_est

	@save "$(folder)/perp1_list"  perp1_list
	@save "$(folder)/perp2_list"  perp2_list

end

main(ARGS)
