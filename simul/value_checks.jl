mean(norm_grad_results)
median(norm_grad_results)


est_theta = [γ[i] ./ sum(γ[i]) for i in 1:N]
est_theta1 = [[sum(est_theta[i][k1, :]) for k1 in 1:K1] for i in 1:N]
est_theta2 = [[sum(est_theta[i][:,k2]) for k2 in 1:K2] for i in 1:N]
# mean([mean(abs.((est_theta1[1] .- Theta1[1])./Theta1[1])) for i in 1:N])
# mean([mean(abs.((est_theta2[1] .- Theta2[1])./Theta2[1])) for i in 1:N])
# mean([mean(abs.((est_theta[1] .- Theta[1])./Theta[1])) for i in 1:N])

# Plots.histogram(norm_grad_results)
describe(norm_grad_results)
# Plots.heatmap(Alpha)
i=200
[compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])
for k1 in 1:K1 for k2 in 1:K2]
norms = [compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])[1]
for k1 in 1:K1 for k2 in 1:K2]
sum(norms.^2)/length(norms)


# compute_∇ℒ_γ_atom(K1, K2, 1, 6,Alpha,γ[10],phi1[10], phi2[10])

Plots.heatmap(Theta[1])
Plots.heatmap(est_theta[1])

i = 4
compute_∇ℒ_γ_atom(K1, K2, k1, k2,Alpha,γ[i],phi1[i], phi2[i])
println(Theta1[i]);println(est_theta1[i]);
println(Theta2[i]);println(est_theta2[i]);
println(Theta[i]);println(est_theta[i]);


compute_ℒ_γ_atom(K1, K2, 1, 2, Alpha, γ[i], phi1[i], phi2[i])
# 599/(360000*6)

trigamma_(sum(γ[1]))
size(phi1[1],1)+size(phi2[1],1)+sum(Alpha)
sum(γ[1])
describe([sum(γ[i]) for i in 1:N])
