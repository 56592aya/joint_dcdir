include("utils.jl")
function create_Alpha(K1, K2)
    alpha_vec = Float64[]
    while length(alpha_vec) < K1*K2
        r = rand(Normal(1.0/(K1*K2), 1.0/(K1*K2)^2))
        # r = rand(1)[1]
        if r > 0
            alpha_vec = vcat(alpha_vec, r)
        end
    end
    Alpha = convert(Matrix{Float64}, transpose(reshape(alpha_vec, (K2, K1))))
    return alpha_vec, Alpha
end


function create_Theta(alpha_vec, N, K1, K2)
    Theta_vec = convert(Matrix{Float64},transpose(rand(Distributions.Dirichlet(alpha_vec),N)))
    Theta = [convert(Matrix{Float64},transpose(reshape(Theta_vec[i,:], (K2, K1)))) for i in 1:N]
    Theta1 = [sum(Theta[i], dims=2)[:,1] for i in 1:N]
    Theta2 = [sum(Theta[i], dims=1)[1,:] for i in 1:N]
    return Theta_vec, Theta, Theta1, Theta2
end

function create_phis!(phi, Thetax, N)
    for i in 1:N
		wlen, K = size(phi[i])
        for w in 1:wlen
            s = sum(rand(Distributions.Multinomial(1,Thetax[i]), 100), dims=2)[:,1] .*1.0
            s ./= sum(s)
            phi[i][w,:] = s
        end
    end
    return phi
end


K1 = 10
K2 = 12
N = 1000
alpha_vec, Alpha = create_Alpha(K1, K2)
Theta_vec, Theta, Theta1, Theta2 = create_Theta(alpha_vec, N, K1, K2)
@assert sum([sum(Theta[i]) for i in 1:N]) == sum([sum(Theta2[i]) for i in 1:N]) == sum([sum(Theta1[i]) for i in 1:N]) == N*1.0
# fixed_len1 = 50
# fixed_len2 = 50
# fixed_len1 = Int64.(ceil.(50 .+ (950) .* rand(N)))
# fixed_len2 = Int64.(ceil.(50 .+ (950) .* rand(N)))
fixed_len1 = Int64.(ceil.(rand(Exponential(.1), 1000).*1000))
fixed_len2 = Int64.(ceil.(rand(Exponential(.08), 1000).*1000))
phi1 = [zeros(Float64, (fixed_len1[i],K1)) for i in 1:N]
phi2 = [zeros(Float64, (fixed_len2[i],K2)) for i in 1:N]
create_phis!(phi1, Theta1, N)
create_phis!(phi2, Theta2, N)

γ = [Alpha for i in 1:N]

# function compute_ℒ_γ(N, K1, K2, Alpha, γ, fixed_len1, fixed_len2, phi1, phi2)
#     ℒ_γ = zeros(Float64, N)
#     ℒ_γ[i] -= lgamma_(sum(γ))
#     for i in 1:N
#         for k1 in 1:K1
#             for k2 in 1:K2
#                 ℒ_γ[i] += (Alpha[k1, k2]-γ[i][k1, k2])*(digamma_(γ[i][k1, k2])- digamma_(sum(γ[i])))
#                 ℒ_γ[i] += (lgamma_(γ[i][k1, k2]))
#
#             end
#         end
#         for k1 in 1:K1
#             for w in 1:fixed_len1[i]
#                 ℒ_γ[i] += phi1[i][w,k1]*(digamma_(sum(γ[i][k1,:]) - digamma_(sum(γ[i]))))
#             end
#         end
#         for k2 in 1:K2
#             for w in 1:fixed_len2[i]
#                 ℒ_γ[i] += phi2[i][w,k2]*(digamma_(sum(γ[i][:,k2]) - digamma_(sum(γ[i]))))
#             end
#         end
#     end
#     return ℒ_γ
# end

function compute_ℒ_γ_atom(K1, K2, k1, k2, Alpha, γ, phi1, phi2)##gamma  and phis are i-indexed
    ℒ_γ = 0.0
    ℒ_γ += (Alpha[k1, k2]-γ[k1, k2])*(digamma_(γ[k1, k2]))
    for l1 in 1:K1
        for l2 in 1:K2
            ℒ_γ -= (Alpha[l1, l2]-γ[l1, l2])*digamma_(sum(γ))
        end
    end
    ℒ_γ -= lgamma_(sum(γ))
    ℒ_γ += (lgamma_(γ[k1, k2]))

    for w in 1:size(phi1,1)
        ℒ_γ += phi1[w,k1] * (digamma_(sum(γ[k1,:])))
        for l1 in 1:K1
            ℒ_γ -= phi1[w,l1] * digamma_(sum(γ))
        end
    end
    for w in 1:size(phi2,1)
        ℒ_γ += phi2[w,k2]*(digamma_(sum(γ[:,k2])))
        for l2 in 1:K2
            ℒ_γ -= phi2[w,l2] * digamma_(sum(γ))
        end
    end
    return ℒ_γ
end


 # compute_ℒ_γ_atom(K1, K2, k1, k2,
 #  Alpha, γ[i], fixed_len1, fixed_len2, phi1[i], phi2[i])

function compute_∇ℒ_γ_atom(K1, K2, k1, k2,γ,phi1, phi2) # indexed at i
    rest = 0.0
    rest += (Alpha[k1, k2]-γ[k1, k2])*(trigamma_(γ[k1, k2]))
    for w in 1:size(phi1,1)
        rest += phi1[w,k1]*trigamma_(sum(γ[k1,:]))
    end
    for w in 1:size(phi2,1)
        rest += phi2[w,k2]*trigamma_(sum(γ[:,k2]))
    end

    special_term = 0.0
    for l1 in 1:K1
        for l2 in 1:K2
            special_term += (Alpha[l1, l2]-γ[l1, l2])
        end
    end
    special_term += size(phi1,1)+size(phi2,1)
	special_term *= trigamma_(sum(γ))
    ∇_γ = rest + special_term

    return ∇_γ , rest , special_term
end


function optimize_γ_i(K1, K2,old, Alpha,γ,phi1, phi2, ITER)
    γ_running = old
	for iter in 1:ITER
		for k1 in 1:K1
			for k2 in 1:K2
				sum1 = sum([phi1[w,k1] for w in 1:size(phi1,1)])
				sum2 = sum([phi2[w,k2] for w in 1:size(phi2,1)])
				γ_running[k1,k2] = Alpha[k1, k2] + sum1*(γ_running[k1, k2]/sum(γ_running[k1,:]))
				γ_running[k1,k2] = Alpha[k1, k2] + sum2*(γ_running[k1, k2]/sum(γ_running[:,k2]))
			end
		end
	end
	return γ_running
end

prior_grad_results = zeros(Float64, N)
for i in 1:N
	prior_grad_results[i] = mean([compute_∇ℒ_γ_atom(K1, K2, k1, k2,γ[i],phi1[i], phi2[i])[1]
	 for k1 in 1:K1 for k2 in 1:K2])
end
Plots.histogram(prior_grad_results)
grad_results = zeros(Float64, N)
for i in 1:N
	if i % 10 == 0
		println(i)
	end
	γ[i] = optimize_γ_i(K1, K2,Alpha, Alpha,γ[i], phi1[i], phi2[i], 100)

	grad_results[i] = mean([compute_∇ℒ_γ_atom(K1, K2, k1, k2,γ[i],phi1[i], phi2[i])[1]
	 for k1 in 1:K1 for k2 in 1:K2])
end
mean(grad_results)
median(grad_results)

Plots.histogram(grad_results)
