include("test_opt.jl")
# N, K1, K2, alpha_vec, Alpha,Theta_vec, Theta, Theta1, Theta2,
# fixed_len1, fixed_len2,phi1, phi2, γ = init_param(5, 5, 6, 2000, 2000)
N, K1, K2, alpha_vec, Alpha,Theta_vec, Theta, Theta1, Theta2,
fixed_len1, fixed_len2,phi1, phi2, γ = init_param_2(5, 5, 6, 2000, 2000)

# norm_grad_results = update_γ(10, N, K1, K2, γ, Alpha, phi1, phi2)
norm_grad_results = update_γ_2(1, N, K1, K2, γ, Alpha, phi1, phi2)
describe(norm_grad_results)
mean(norm_grad_results)
median(norm_grad_results)


display([sum([sum(γ[i][k1,:]) for k1 in 1:K1]) for i in 1:N])
display([sum(γ[1][k1,:]) for k1 in 1:K1])


# display([sum(phi1[1][:,k1]) for k1 in 1:K1])

sum(Alpha)
display(sum(Alpha, dims=2)[:,1])
i=1
# display([compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i]) for k1 in 1:K1 for k2 in 1:K2])
display([compute_∇ℒ_γ_atom_2(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i]) for k1 in 1:K1 for k2 in 1:K2])
display([compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],sum(phi1[i], dims=3)[:,:,1], sum(phi2[i], dims=2)[:,1,:]) for k1 in 1:K1 for k2 in 1:K2])





# est_theta = [γ[i] ./ sum(γ[i]) for i in 1:N]
# est_theta1 = [[sum(est_theta[i][k1, :]) for k1 in 1:K1] for i in 1:N]
# est_theta2 = [[sum(est_theta[i][:,k2]) for k2 in 1:K2] for i in 1:N]
Plots.heatmap(Theta[1])
Plots.heatmap(est_theta[1])

i = 4
compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])
display(Theta1[i]);println(est_theta1[i]);
display(Theta2[i]);println(est_theta2[i]);
display(Theta[i]);println(est_theta[i]);


compute_ℒ_γ_atom(K1, K2, 1, 2, Alpha, γ[i], phi1[i], phi2[i])

trigamma_(sum(γ[1]))
size(phi1[1],1)+size(phi2[1],1)+sum(Alpha)
sum(γ[1])
describe([sum(γ[i]) for i in 1:N])
describe([sum(γ[i]) for i in 1:N])
