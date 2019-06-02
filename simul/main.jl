include("utils.jl")
include("funcs.jl")
include("dgp_simul.jl")

function main()
	MAX_ITER = 100
    N, K1, K2, wlens1, wlens2,alpha_vec, Alpha,beta1, beta2,phi1, phi2, γ, b1, b2 =
	init_params(K1, K2, .09, .08,0.002, corp1, corp2,V1, V2);
	for iter in 1: MAX_ITER
		for i in 1:N
	 		doc1 = corp1[i]
	 		doc2 = corp2[i]
	 		for (w, val) in enumerate(doc1)
	 			phi1[i][w,:, :] = optimize_phi_iw!(phi1[i], γ[i],b2, K1, K2, V1, w, doc1, 1)
	 		end
			phi1
	 		for (w,val) in enumerate(doc2)
	 			phi2[i][w,:, :] = optimize_phi_iw!(phi2[i], γ[i],b2, K1, K2, V2, w, doc2, 2)
	 		end
	 	end
	 	println(iter)

	 	optimize_γ!(N, K1, K2, Alpha,γ, phi1, phi2)
	 	for k in 1:K1
	 		optimize_b_per_topic!(N, b1, beta1, k, phi1, 1, corp1, V1)
	 	end
	 	for k in 1:K2
	 		optimize_b_per_topic!(N, b2, beta2, k, phi2, 2, corp2, V2)
	 	end
	 	if (iter % 10 == 0) || (iter == 1)
	 		println(compute_ℒ_full(N,K1,K2,V1,V2,beta1,beta2,b1,b2,
									Alpha,γ,corp1,corp2,phi1,phi2))
	 	end
	end
	# estimate_thetas(γ)
end
main()
