include("utils.jl")
include("funcs.jl")
include("dgp_simul.jl")

function main()
	ELBO = Float64[]
	MAX_ITER = 10000
    N, K1, K2, wlens1, wlens2,alpha_vec, Alpha,beta1, beta2,phi1, phi2, γ, b1, b2, Elog_B1, Elog_B2, Elog_Theta =
	init_params(K1, K2, .5, .5, .5, corp1, corp2,V1, V2);


	# beta1 = Β1_truth
	# beta2 = Β2_truth
	# Alpha = Α_truth
	# alpha_vec = α_truth
	update_Elogtheta!(γ, Elog_Theta)

	update_Elogb!(b1, Elog_B1)
	update_Elogb!(b2, Elog_B2)
	γ_old = deepcopy(γ)

	for iter in 1:MAX_ITER
		global ELBO
		for i in 1:N
			γ[i] =  ones(Float64, (K1, K2))
			Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])
	 		doc1 = corp1[i]
	 		doc2 = corp2[i]
			for _u in 1:5
		 		for (w, val) in enumerate(doc1)
		 			phi1[i][w,:, :] .= optimize_phi1_iw(phi1[i], Elog_Theta[i,:,:],Elog_B1, K1, K2, w, doc1)
		 		end
		 		for (w,val) in enumerate(doc2)
		 			phi2[i][w,:, :] .= optimize_phi2_iw(phi2[i], Elog_Theta[i,:,:],Elog_B2, K1, K2, w, doc2)
		 		end
				γ_old[i] .= deepcopy(γ[i])
				optimize_γi!(K1, K2, Alpha,γ[i], phi1[i], phi2[i])
				Elog_Theta[i,:,:] = update_Elogtheta_i(γ[i], Elog_Theta[i,:,:])

				# if gamma_converged(γ[i], γ_old[i])
				# 	println("$i converged")
				# 	break;
				# end
		 	end
		end

		# println(iter)
 		b1 .= optimize_b1(N, beta1, phi1, corp1, K1, V1)
		update_Elogb!(b1, Elog_B1)
 		b2 .= optimize_b2(N, beta2, phi2, corp2, K2, V2)
		update_Elogb!(b2, Elog_B2)
	 	if (iter % 5 == 0) || (iter == 1)
			elbo = compute_ℒ_full(N,K1,K2,V1,V2,beta1,beta2,b1,b2,
									Alpha,γ,corp1,corp2,phi1,phi2)
			ELBO = vcat(ELBO, elbo)
	 		println(elbo)
			println(iter)

			if div(iter,5) > 50
				elb_idx =  div(iter,5)
				if ((ELBO[elb_idx] - ELBO[elb_idx-1]) < 0.0) && ((ELBO[elb_idx] - ELBO[elb_idx-2]) < 0.0) &&
					((ELBO[elb_idx] - ELBO[elb_idx-3]) < 0.0)
					break;
				end
				converged = (abs((ELBO[elb_idx] - ELBO[elb_idx-1])/ELBO[elb_idx-1]) < 1e-12) &&
				(abs((ELBO[elb_idx-1] - ELBO[elb_idx-2])/ELBO[elb_idx-2]) < 1e-12)
				if converged
					break;
				end
			end
			# (ELBO[115]-ELBO[114])/ELBO[114]
	 	end
	end
	theta_est = estimate_thetas(γ)
	Plots.plot(collect(1:length(ELBO)), ELBO)
	png("ELBO")
	B1_est = estimate_B(b1)
	B2_est = estimate_B(b2)
	theta_est[1]

	Plots.heatmap(theta_est[68][[2,3,1,5,4],[3,1,2,5,4]], yflip = true)
	# png("theta_est_871")
	Plots.heatmap(Θ_truth[68], yflip = true)
	# png("theta_true_871")

	# Plots.heatmap(Alpha, yflip = true)
	# count_V1 = zeros(Int64, V1)
	# count_V2 = zeros(Int64, V2)
	# for i in 1:N
	# 	for (w,val) in enumerate(corp1[i])
	# 		count_V1[val] += 1
	# 	end
	# 	for (w,val) in enumerate(corp2[i])
	# 		count_V2[val] += 1
	# 	end
	# end
	# Plots.histogram(count_V1)
	# Plots.histogram(count_V2)
	Plots.heatmap(B1_est, yflip = true)
	# png("B1_est_unlabeled")
	Plots.heatmap(B1_est[[2,3,1,5,4],:], yflip = true)
	# png("B1_est_labeled")
	Plots.heatmap(Β1_truth, yflip = true)
	Plots.heatmap(B2_est, yflip = true)
	# png("B2_est_unlabeled")
	Plots.heatmap(B2_est[[3,1,2,5,4],:], yflip = true)
	# png("B2_est_labeled")
	Plots.heatmap(Β2_truth, yflip = true)


end
main()
