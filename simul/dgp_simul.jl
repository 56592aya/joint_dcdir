
N, K1, K2, V1, V2 = 200, 50, 50, 1000, 1000
β1_single_truth, β2_single_truth = 0.1, 0.1
wlen1_single, wlen2_single = 100, 100

α_truth,Α_truth, θ_truth,Θ_truth,
 Β1_truth, Β2_truth, β1_truth, β2_truth,
  V1, V2, corp1, corp2 = Create_Truth(N, K1, K2, V1, V2, β1_single_truth, β2_single_truth, wlen1_single, wlen2_single)


  #
  # y1 = Int64[]
  # y2 = Int64[]
  # for i in 1:N
  # 	global y1,y2
  # 	y1 = vcat(y1, corp1[i])
  # 	y1 = unique(y1)
  # 	y2 = vcat(y2, corp2[i])
  # 	y2 = unique(y2)
  # end
  # intersect(corp1[3],corp1[4])
  # y1
  # y2
