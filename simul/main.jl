include("utils.jl")
include("funcs.jl")
include("dgp_simul.jl")

function main()
	MAX_ITER = 2000
    N, K1, K2, wlens1, wlens2,alpha_vec, Alpha,beta1, beta2,phi1, phi2, γ, b1, b2, Elog_B1, Elog_B2, Elog_Theta =
	init_params(K1, K2, .3, .3,0.01, corp1, corp2,V1, V2);
	update_Elogtheta!(γ, Elog_Theta)
	update_Elogb!(b1, Elog_B1)
	update_Elogb!(b2, Elog_B2)
	for iter in 1:MAX_ITER
		for i in 1:N
	 		doc1 = corp1[i]
	 		doc2 = corp2[i]
	 		for (w, val) in enumerate(doc1)
	 			phi1[i][w,:, :] = optimize_phi1_iw!(phi1[i], Elog_Theta[i,:,:],Elog_B1, K1, K2, w, doc1)
	 		end
			phi1[i]
	 		for (w,val) in enumerate(doc2)
	 			phi2[i][w,:, :] = optimize_phi2_iw!(phi2[i], Elog_Theta[i,:,:],Elog_B2, K1, K2, w, doc2)
	 		end
			phi2[i]
	 	end

	 	println(iter)
		x = deepcopy(γ)
	 	γ = optimize_γ!(N, K1, K2, Alpha,γ, phi1, phi2)

		update_Elogtheta!(γ, Elog_Theta)
	 	for k in 1:K1
	 		optimize_b1_per_topic!(N, b1, beta1, k, phi1, corp1, V1)
	 	end
		update_Elogb!(b1, Elog_B1)
	 	for k in 1:K2
	 		optimize_b2_per_topic!(N, b2, beta2, k, phi2, corp2, V2)
	 	end
		update_Elogb!(b2, Elog_B2)
	 	if (iter % 20 == 0) || (iter == 1)
	 		println(compute_ℒ_full(N,K1,K2,V1,V2,beta1,beta2,b1,b2,
									Alpha,γ,corp1,corp2,phi1,phi2))
	 	end
	end
	theta_est = estimate_thetas(γ)
	phi1[1] .- phi1[2]
	Θ_truth
	estimate_B(b1)
	estimate_B(b2)

	Plots.heatmap(theta_est[2])
	Plots.heatmap(Θ_truth[16])
	Plots.heatmap(estimate_B(b2))
	Plots.heatmap(Β1_truth)
end
main()
