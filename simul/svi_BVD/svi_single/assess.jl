using Plots, FileIO, JLD2
function sort_by_argmax!(X::Matrix{Float64})

	n_row=size(X,1)
	n_col = size(X,2)
	ind_max=zeros(Int64, n_row)
	permuted_index = zeros(Int64, n_row)
	for a in 1:n_row
    	ind_max[a] = findmax(view(X,a,1:n_col))[2]
	end
	X_tmp = similar(X)
	count_ = 1
	for i in 1:n_row
	for j in 1:maximum(ind_max)
    		if ind_max[i] == j
	      		for k in 1:n_col
	        		X_tmp[count_, k] = X[i,k]
	      		end
				permuted_index[count_]=i
      			count_ += 1
    		end
  		end
	end
	# This way of assignment is important in arrays, el by el
	X[:]=X_tmp[:]
	X, permuted_index
end
@load "../truth" Truth_Params
mkdir("pics")
# eee = 1560
# @load "B1_at_epoch_$(eee)" B1_est
# @load "B2_at_epoch_$(eee)" B2_est
# @load "theta_at_epoch_$(eee)" theta_est

@load "B1_last" B1_est
@load "B2_last" B2_est
@load "theta_last" theta_est

B1_truth = deepcopy(Truth_Params.Β1);
B2_truth = deepcopy(Truth_Params.Β2);
theta_truth = deepcopy(Truth_Params.Θ);
l = [1, 2, 3, 4, 5]
m = 100000.0
for i in 1:5
	global m
	for j in 1:5
		val = sum( (B1_truth[i,:] .- B1_est[j,:]).^2)
		if val < m
			m = val
			l[i] = j
		end
	end
	m=100000.0
end
inds1 = deepcopy(l)
l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
m = 100000.0
for i in 1:10
	global m
	for j in 1:10
		val = sum( (B2_truth[i,:] .- B2_est[j,:]).^2)
		if val < m
			m = val
			l[i] = j
		end
	end
	m=100000.0
end
inds2 = deepcopy(l)


p1b = Plots.heatmap(B1_truth, yflip=true)
p1a = Plots.heatmap(B1_est[inds1,:], yflip=true)
p2b = Plots.heatmap(B2_truth, yflip=true)
p2a = Plots.heatmap(B2_est[inds2,:], yflip=true)
plot(p1a, p1b, p2a, p2b, layout =(2, 2), legend=false)
savefig("pics/betas.png")
Plots.heatmap(theta_truth[3], yflip=true)
Plots.heatmap(theta_est[3][inds1, inds2], yflip=true)
i=1;p1=Plots.scatter(B1_truth[i,:], B1_est[inds1[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=1;pp1=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=2;p2=Plots.scatter(B1_truth[i,:], B1_est[inds1[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=2;pp2=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:],grid=false, aspect_ratio=:equal,legend=false);
i=3;p3=Plots.scatter(B1_truth[i,:], B1_est[inds1[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=3;pp3=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=4;p4=Plots.scatter(B1_truth[i,:], B1_est[inds1[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=4;pp4=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=5;p5=Plots.scatter(B1_truth[i,:], B1_est[inds1[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=5;pp5=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=6;pp6=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=7;pp7=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=8;pp8=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=9;pp9=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);
i=10;pp10=Plots.scatter(B2_truth[i,:], B2_est[inds2[i],:], grid=false, aspect_ratio=:equal,legend=false);



plot(p1, pp1, p2, pp2, p3, pp3, p4, pp4, p5, pp5, layout =(2, 5), legend=false)

theta_truth_1 = zeros(Float64, (length(theta_truth), 5))
for i in 1:size(theta_truth_1,1)
	for j in 1:5
		theta_truth_1[i,j] = sum(theta_truth[i][j,:])
	end
end
theta_truth_2 = zeros(Float64, (length(theta_truth), 10))
for i in 1:size(theta_truth_2,1)
	for j in 1:10
		theta_truth_2[i,j] = sum(theta_truth[i][:,j])
	end
end

theta_est_1 = zeros(Float64, (length(theta_truth), 5))
theta_est_2 = zeros(Float64, (length(theta_truth), 10))

for i in 1:size(theta_est_1,1)
	for j in 1:5
		theta_est_1[i,j] = sum(theta_est[i][inds1[j],:])
	end
end
for i in 1:size(theta_est_2,1)
	for j in 1:10
		theta_est_2[i,j] = sum(theta_est[i][:,inds2[j]])
	end
end
x = collect(range(0.0, 1.0, length=100))
y = collect(range(0.0, 1.0, length=100))

i = 1;scatter(theta_truth_1[:,i], theta_est_1[:,i], grid=false, aspect_ratio=:equal,legend=false);plot!(x, y, linewidth=3);
savefig("pics/theta1_$(i).png")
i = 2;scatter(theta_truth_1[:,i], theta_est_1[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta1_$(i).png")
i = 3;scatter(theta_truth_1[:,i], theta_est_1[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta1_$(i).png")
i = 4;scatter(theta_truth_1[:,i], theta_est_1[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta1_$(i).png")
i = 5;scatter(theta_truth_1[:,i], theta_est_1[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta1_$(i).png")
i = 1;scatter(theta_truth_2[:,i], theta_est_2[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta2_$(i).png")
i = 2;scatter(theta_truth_2[:,i], theta_est_2[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta2_$(i).png")
i = 3;scatter(theta_truth_2[:,i], theta_est_2[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
 savefig("pics/theta2_$(i).png")
i = 4;scatter(theta_truth_2[:,i], theta_est_2[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta2_$(i).png")
i = 5;scatter(theta_truth_2[:,i], theta_est_2[:,i], grid=false, aspect_ratio=:equal,legend=false);
plot!(x, y, linewidth=3);
savefig("pics/theta2_$(i).png")

sum([(1024+i)^(-.7) for i in 1:500000])
1/16
16*128
1/40
0.000194*1600
1/78
(1024+1)^(-.7)
1/5
