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
		"-m"               #number of communities
            help = "which mode"
            arg_type=Int64
            default=1
		"--data"            #data folder
            arg_type = String
            required = true
		"-k"               #number of communities
            help = "number of topics"
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
		"--beta_prior"
			help = "beta prior"
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
	dd = parsed_args["data"]
	global data_folder = "../$(dd)"
	global K = parsed_args["k"]
	global m = parsed_args["m"]
	global α_single_prior = parsed_args["alpha_prior"]
	global β_single_prior = parsed_args["beta_prior"]
	global S = parsed_args["S"]
	global κ = parsed_args["kappa"]
	global every = parsed_args["every"]
	global MAXITER = parsed_args["maxiter"]
	global mb_size = parsed_args["mbsize"]
	global h = parsed_args["holdout"]



	 # global m = 1
	 # global K = 5
 	 # global α_single_prior = .3
 	 # global β_single_prior = .3
	 # global S = 256.0
	 # global κ = .7
	 # global every = 5
	 # global MAXITER = 500000
	 # global mb_size = 128
	 # global h = 0.005
	 # data_folder = "../5000_5_10_200_400_0.2_0.1_0.3"

	global folder = mkdir("$(data_folder)/single_est_m$(m)_$(K)_$(mb_size)_$(MAXITER)_$(h)_$(S)_$(κ)_$(every)_$(α_single_prior)_$(β_single_prior)")
#########################

	@load "$(data_folder)/corpus1" Corpus1
	@load "$(data_folder)/corpus2" Corpus2
	corpus = m==1 ? deepcopy(Corpus1) : deepcopy(Corpus2)
	global N = corpus.N

	alpha_vec, Alpha,beta,phi, γ, b, Elog_B, Elog_Theta =
				init_params(K, β_single_prior,  α_single_prior, corpus);

	update_Elogtheta!(γ, Elog_Theta)
	update_Elogb!(b, Elog_B)
	h_map = setup_train_test(h, N, corpus)

	@save "$(folder)/h_map" h_map
	mbs, nb = epoch_batches(N, mb_size, h_map)

	mindex, epoch_count = 1,0
	count_params = CountParams(corpus.N-sum(h_map), K)
	ho, obs, w_in_phi,w_in_ho =create_test(h_map, corpus)
	perp_list = Float64[]

	@info "VI Started"
	global converged = false
	for iter in 1:MAXITER

		# global mindex, nb, mbs, count_params,mb_size, perp_list,epoch_count

		if mindex == (nb+1)
			mbs, nb = epoch_batches(N, mb_size, h_map)
			mindex = 1
			theta_est = estimate_thetas(γ)
			B_est = estimate_B(b)
			if epoch_count % every == 0
				@info "starting to calc perp"
				p = calc_perp(obs,ho,corpus, γ, Alpha, Elog_Theta,
				 Elog_B, count_params, phi, w_in_phi, w_in_ho, B_est)
				 perp_list = vcat(perp_list, p)
				 @info "perp=$(p)"
				@save "$(folder)/theta_at_epoch_$(epoch_count)"  theta_est
				@save "$(folder)/B_at_epoch_$(epoch_count)"  B_est
				if length(perp_list) > 2
					if (abs(perp_list[end]-perp_list[end-1])/perp_list[end] < 1e-8)
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
		sum_phi_mb = zeros(Float64, (count_params.K,corpus.V))
		sum_phi_i = zeros(Float64, count_params.K)
		for i in mb

			Elog_Theta[i,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:])
	 		doc = deepcopy(corpus.Data[i])
			γ_old = deepcopy(γ[i])
			gamma_c = false

			for _u in 1:20
				sum_phi_i = zeros(Float64, count_params.K)
				for val in unique(doc)

					x = findall(x -> x == val, doc)
					y = optimize_phi_iw(Elog_Theta[i,:],Elog_B, count_params,val)
					for xx in x
						phi[i][xx,:] .= y
					end

					sum_phi_i .+= length(x).*y
					if (_u == 20) || gamma_c
						sum_phi_mb[:,val] .+= length(x).* y
					end
				end

				optimize_γi!(count_params.K, Alpha,γ[i], sum_phi_i)
				Elog_Theta[i,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:])
				if (mean(abs.(γ_old .- γ[i])/γ[i])) < 1e-3
					gamma_c = true
				end
				γ_old = deepcopy(γ[i])
	 		end

		end # i in mb end


		old_b = deepcopy(b)
		b .= optimize_b(length(mb), beta, sum_phi_mb, corpus.V, count_params.K, count_params.N)
		b .= (1-ρ).*old_b + ρ.*b
		update_Elogb!(b, Elog_B)

		mindex += 1
							  ################################
									###For FINAL Rounds###
							  ################################
		if iter == MAXITER || converged
			@info "Final rounds"
			mb = collect(1:N)[.!h_map]
			sum_phi_mb = zeros(Float64, (count_params.K,corpus.V))
			sum_phi_i = zeros(Float64, count_params.K)
			for i in mb
				Elog_Theta[i,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:])
		 		doc = deepcopy(corpus.Data[i])
				γ_old = deepcopy(γ[i])
				for _u in 1:30
					sum_phi_i = zeros(Float64, count_params.K)
					for val in unique(doc)

						x = findall(x -> x == val, doc)
						y = optimize_phi_iw(Elog_Theta[i,:],Elog_B, count_params,val)
						for xx in x
							phi[i][xx,:] .= y
						end

						sum_phi_i .+= length(x).*y
						if (_u == 30) || gamma_c
							sum_phi_mb[:,val] .+= length(x).* y
						end
					end
					optimize_γi!(count_params.K, Alpha,γ[i], sum_phi_i)
					Elog_Theta[i,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:])
		 		end
			end
			b .= optimize_b(length(mb), beta, sum_phi_mb, corpus.V, count_params.K, count_params.N)
			update_Elogb!(b, Elog_B)
			break
		end

	end
	theta_est = estimate_thetas(γ)
	@save "$(folder)/theta_last"  theta_est
	B1_est = estimate_B(b1)
	@save "$(folder)/B_last"  B_est

	@save "$(folder)/perp_list"  perp_list

end

main(ARGS)
