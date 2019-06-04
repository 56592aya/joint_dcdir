include("utils.jl")
include("funcs.jl")
include("dgp_simul.jl")
function main()
	ELBO = Float64[]
	MAX_ITER = 10000
    N, K1, K2, wlens1, wlens2,alpha_vec, Alpha,beta1, beta2,phi1, phi2, γ, b1, b2, Elog_B1, Elog_B2, Elog_Theta =
	init_params(K1, K2, .08, .07, 1.0, corp1, corp2,V1, V2);
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
		minimum(b1)
 		b1 .= optimize_b1(N, beta1, phi1, corp1, V1, K1)
		update_Elogb!(b1, Elog_B1)
		minimum(Elog_B1)
 		b2 .= optimize_b2(N, beta2, phi2, corp2, V2, K2)
		update_Elogb!(b2, Elog_B2)
		minimum(Elog_B2)
	 	if (iter % 5 == 0) || (iter == 1)
			elbo = compute_ℒ_full(N,K1,K2,V1,V2,beta1,beta2,b1,b2,
									Alpha,γ,corp1,corp2,phi1,phi2)
			ELBO = vcat(ELBO, elbo)
	 		println(elbo)
			println(iter)
			# theta_est = estimate_thetas(γ)
			# Plots.heatmap(theta_est[1])
			# png("$(iter).png")
	 	end
	end
	theta_est = estimate_thetas(γ)
	Plots.plot(collect(1:length(ELBO)), ELBO)
	B1_est = estimate_B(b1)
	B2_est = estimate_B(b2)
	theta_est[1]
	Plots.heatmap(theta_est[12], yflip = true)
	theta_est[1]
	Plots.heatmap(Θ_truth[12], yflip = true)
	Plots.heatmap(Alpha, yflip = true)
	theta_est[1] .- theta_est[3]
	Plots.heatmap(Θ_truth[1], yflip = true)
	Plots.heatmap(B1_est, yflip = true)
	Plots.heatmap(B2_est, yflip = true)
	Plots.heatmap(Β2_truth, yflip = true)
	Β2_truth
end
main()
