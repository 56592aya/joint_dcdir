
N, K1, K2, V1, V2 = 2000, 5, 5, 200, 200
β1_single_truth, β2_single_truth = .3, .3
wlen1_single, wlen2_single = 200, 200

# α_truth,Α_truth, θ_truth,Θ_truth,
#  Β1_truth, Β2_truth, β1_truth, β2_truth,
#   V1, V2, corp1, corp2
  # = Create_Truth(N, K1, K2, V1, V2, β1_single_truth, β2_single_truth, wlen1_single, wlen2_single)


function simulate_data(N, K1, K2, V1, V2,β1_single_truth, β2_single_truth,wlen1_single, wlen2_single)
    y1 = Int64[]
	y2 = Int64[]
    while true
        α_truth,Α_truth, θ_truth,Θ_truth,
		Β1_truth, Β2_truth, β1_truth, β2_truth,V1, V2, corp1, corp2 =
		Create_Truth(N, K1, K2, V1, V2, β1_single_truth, β2_single_truth, wlen1_single, wlen2_single)
		for i in 1:N
            y1 = unique(y1)
		  	y2 = unique(y2)
		    y1 = vcat(y1, corp1[i])
		    y2 = vcat(y2, corp2[i])
		end
		y1 = unique(y1)
		y2 = unique(y2)
        println(length(y1))
        println(length(y2))
		if ((length(y1) == V1) && (length(y2) == V2))
            println(length(y1))
		    println(length(y2))
            return α_truth,Α_truth, θ_truth,Θ_truth,Β1_truth, Β2_truth, β1_truth, β2_truth,V1, V2, corp1, corp2
		else
            y1 = Int64[]
        	y2 = Int64[]
        end
    end
end

α_truth,Α_truth, θ_truth,Θ_truth,Β1_truth, Β2_truth, β1_truth, β2_truth,V1, V2, corp1, corp2=
simulate_data(N, K1, K2, V1, V2,β1_single_truth, β2_single_truth,wlen1_single, wlen2_single)


VV1 = Int64[]
VV2 = Int64[]
for i in 1:N
  global VV1,VV2
  VV1 = unique(VV1)
  VV2 = unique(VV2)
  VV1 = vcat(VV1, corp1[i])
  VV2 = vcat(VV2, corp2[i])
end

VV1 = unique(VV1)
VV2 = unique(VV2)

BB1 = deepcopy(Β1_truth)
BB2 = deepcopy(Β2_truth)
Plots.heatmap(Β1_truth, yflip = true)
# png("B1_unsorted")
Plots.heatmap(sort_by_argmax!(convert(Matrix{Float64},transpose(BB1)))[1], yflip=true)
# png("B1_sorted")
Plots.heatmap(Β2_truth, yflip = true)
# png("B2_unsorted")
Plots.heatmap(sort_by_argmax!(convert(Matrix{Float64},transpose(BB2)))[1], yflip=true)
# png("B2_sorted")
