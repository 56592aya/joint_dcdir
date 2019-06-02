




# function update_γ(N_, K1_, K2_, γ_, Alpha_, phi1_, phi2_)
# 	"""
# 	updating all individual gammas
# 	updates		gamma in-place
# 	Returns		mean of the norm gradients
# 	"""
# 	norm_grad_results = zeros(Float64, N)
# 	for i in 1:N_
# 		γ_[i] = optimize_γ_ind(K1_, K2_, Alpha_,phi1_[i], phi2_[i])
# 		norm_grad_results[i] = mean(norm.([compute_∇ℒ_γ_atom(K1_, K2_, k1, k2,Alpha_,γ_[i],phi1_[i], phi2_[i])[1]
# 		 for k1 in 1:K1_ for k2 in 1:K2_]))
# 	end
# 	return norm_grad_results
# end

function compute_ℒ_b_atom(beta_prior, b, k, K, v, V_, doc, phi, dim)

	dig = digamma_(b[k,v])
	dig_sum = digamma_(sum(b[k,:]))
	lg = SpecialFunctions.lgamma(b[k,v])
	lg_sum = SpecialFunctions.lgamma_(sum(b[k,:]))

	ℒ_b = 0.0
	ℒ_b += (beta_prior[k,v] - b[k,v]) * dig

	for v_ in 1:V_
		ℒ_b -= (beta_prior[k,v] - b[k,v_]) * dig_sum
	end

	ℒ_b += (beta_prior[k,v] - b[k,v_]) * lg

	for i in 1:N
		for (w, val) in enumerate(doc)
			# V_val = parse(Int64, val[5:end])
			if dim == 1
				ℒ_b -= sum(phi[i][w,k, :])*dig_sum
			else
				ℒ_b -= sum(phi[i][w,:, k])*dig_sum
			end
			if val == v
				if dim == 1
					ℒ_b += sum(phi[i][w,k, :])*dig
				else
					ℒ_b += sum(phi[i][w,:, k])*dig
				end
			end
		end
	end
end







# function updates()





	MAX_ITER = 10000


	for iter in 1:MAX_ITER

		for i in 1:N

			doc1 = Corp1[i]
			doc2 = Corp2[i]

			for (w, val) in enumerate(doc1)
				phi1[i][w,:, :] = optimize_phi_iw(phi1[i], γ[i],b1, K1, K2, V1, w,doc1, 1)
			end
			# phi1[i][:,:,:]
			for (w,val) in enumerate(doc2)
				phi2[i][w,:, :] = optimize_phi_iw(phi2[i], γ[i],b2, K1, K2, V2, w, doc2, 2)
			end
			# phi2[i][:,:,:]
		end
		# println(iter)

		optimize_γ!(N, K1, K2, Alpha,γ, phi1, phi2)
		for k in 1:K1
			optimize_b_vec!(N, b1, beta1_prior, k, phi1, 1, Corp1, V1)
		end
		for k in 1:K2
			optimize_b_vec!(N, b2, beta2_prior, k, phi2, 2, Corp2, V2)
		end
		if (iter % 10 == 0) || (iter == 1)
			println(compute_ℒ_full(N,K1,K2,V1,V2,beta1_prior,beta2_prior,b1,b2,
								Alpha,γ,Corp1,Corp2,phi1,phi2))
		end
	end
	theta_est = estimate_thetas(γ)
	B1_est = estimate_B(b1)
	B2_est = estimate_B(b2)
	Plots.heatmap(theta_est[1])
	Plots.heatmap(Theta[1])
# end


# i, k1, k2 = 1, 2, 1
# sum(phi1[i][:,k1,k2]) + sum(phi2[i][:,k1,k2]) + Alpha[k1,k2]


x = [(digamma_(γ[2][k1, k2]) - digamma(sum(γ[2]))) for k1 in 1:K1 for k2 in 1:K2]
y = convert(Matrix{Float64}, transpose(reshape(softmax(x), (K2, K1))))
Plots.heatmap(Theta[2])
Plots.heatmap(y)
sum(theta_est[6], dims=2)[:,1]
sum(y, dims=2)[:,1]
Theta1[1]
sum(theta_est[6], dims=1)[1,:]
Theta1[6]
Theta2[6]
Plots.plot(Beta2[5,:])
Plots.plot(b2[5,:])
