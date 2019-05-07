Base.zero(::Type{KeyVal}) = KeyVal(0,0.0)
function Network(nrows::Int64)

	return spzeros(Int64, nrows,nrows)
end


function digamma_(x::Float64)
	p=zero(Float64)
  x=x+6.0
  p=1.0/(x*x)
  p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
  p=p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
  p
end

function lgamma_(x::Float64)
	z=1.0/(x*x)
 	x=x+6.0
  z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x
  z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1.0)-log(x-2.0)-log(x-3.0)-log(x-4.0)-log(x-5.0)-log(x-6.0)
  z
end

function digamma_(x::Float64, dim::Int64)
    @fastmath @inbounds return sum(digamma_(x+.5*(1-i)) for i in 1:dim)
end


function lgamma_(x::Float64, dim::Int64)
    .25*(dim*dim-1)*log(pi)+sum(lgamma_(x+.5*(1-i)) for i in 1:dim)
end

function logsumexp(X::Vector{Float64})

    alpha = -Inf;r = 0.0;
    @inbounds for x in X
        if x <= alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
    end
    # log1p(r-1.0) + alpha
    log(r) + alpha
end
function logsumexp(X::Float64, Y::Float64)
    alpha = -Inf;r = 0.0;
    if X <= alpha
        r += exp(X - alpha)
    else
        r *= exp(alpha - X)
        r += 1.0
        alpha = X
    end
    if Y <= alpha
        r += exp(Y - alpha)
    else
        r *= exp(alpha - Y)
        r += 1.0
        alpha = Y
    end
    # log1p(r-1.0) + alpha
    log(r) + alpha
end
###Surprisingly the log(a/b) is more stable that log(a) - log(b)
function softmax(X::Vector{Float64})
    lse = logsumexp(X)
    return  @.(exp(X-lse))
end
###Surprisingly the log(a/b) is more stable that log(a) - log(b)
function softmax2(MEM::Vector{Float64},X::Vector{Float64})
    lse = logsumexp(X)
    @.(MEM = (exp(X - lse)))
end

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
	for j in 1:maximum(ind_max)
  		for i in 1:n_row
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
###################INIT#############################
function sort_by_values(v::Vector{KeyVal})
  d = DataFrame()
  d[:X1] = [i for i in 1:length(v)]
  d[:X2] = [0.0 for i in 1:length(v)]
  x = []
  for (i,val) in enumerate(v)
    push!(x,i)
    d[i,:X1] = val.first
    d[i,:X2] = val.second
  end
  x= rand(Float64, 10, 2)

  x = DataFrame(x)

  sort!(d, :X2, rev=true)
  temp = [KeyVal(0,0.0) for z in 1:length(v)]
  for i in 1:length(v)
    temp[i].first = d[i,:X1]
    temp[i].second = d[i,:X2]
  end
  return temp
end


function init_community_mu_init(mu, maxmu, N, topK)
  for i in 1:N
      mu[i,1].first=i
      mu[i,1].second = 1.0 + rand()
      maxmu[i] = i
      for j in 2:topK
        mu[i,j].first = (i+j-1)%N
        mu[i,j].second = rand()
      end
  end
end
function estimate_community_thetas_init(mu,N,topK)
  theta_est = [KeyVal(0,0.0) for i=1:N,j=1:topK]
  for i in 1:N
    s = 0.0
    for k in 1:topK
      s += mu[i,k].second
    end
    for k in 1:topK
      theta_est[i,k].first  = mu[i,k].first
      theta_est[i,k].second = mu[i,k].second*1.0/s
    end
  end
  return theta_est
end
function log_community_groups_init(communities, theta_est, topK, ulinks)
  for link in ulinks
    i = link.first;m=link.second;
    # if i < m
      max_k = 65535
      max = 0.0
      sum = 0.0
      for k1 in 1:topK
        for k2 in 1:topK
          if theta_est[i,k1].first == theta_est[m,k2].first
            u = theta_est[i,k1].second * theta_est[m,k2].second
            sum += u
            if u > max
              max = u
              max_k = theta_est[i,k1].first
            end
          end
        end
      end
      #print("max before is $max and ")
      if sum > 0.0
        max = max/sum
      else
        max = 0.0
      end
      #println(" and max after is $max and sum is $sum")
      if max > .5
        #println(max)
        if max_k != 65535
          i = convert(Int64, i)
          m = convert(Int64, m)
          if haskey(communities, max_k)
            push!(communities[max_k], i)
            push!(communities[max_k], m)
          else
            communities[max_k] = get(communities,max_k,Int64[])
            push!(communities[max_k], i)
            push!(communities[max_k], m)
          end
        end
      end
    # end
  end
  count_ = 1

  # Comm_new = similar(communities)
  Comm_new = Dict{Int64, Vector{Int64}}()
  for c in collect(keys(communities))
    u = collect(communities[c])
    #println(u)
    uniq = Dict{Int64,Bool}()
    ids = Int64[]
    for p in 1:length(u)
      if !(haskey(uniq, u[p]))
        push!(ids, u[p])
        uniq[u[p]] = true
      end
    end
    vids = zeros(Int64,length(ids))

    for j in 1:length(ids)
      vids[j] = ids[j]
    end
    vids = sort(vids)
    if !haskey(Comm_new, count_)
      Comm_new[count_] = get(Comm_new,count_, Int64[])
    end
    for j in 1:length(vids)
      push!(Comm_new[count_], vids[j])
    end
    count_ += 1
  end
  return Comm_new
end
###BatchInfer
## abit dubious on the nested while
function batch_community_infer_init(network::Network, N::Int64)

    topK = 5
    _α = 1.0/N
    mu = [KeyVal(0,0.0) for i=1:N, j=1:topK]
    munext = [Dict{Int64, Int64}() for i in 1:N]
    maxmu = zeros(Int64,N)
    communities = Dict{Int64, Vector{Int64}}()
    ulinks = Vector{Pair{Int64, Int64}}()
    #undirected
    x,y,z=findnz(network)

    for row in 1:nnz(network)
        push!(ulinks,x[row]=>y[row])
    end
    init_community_mu_init(mu, maxmu, N, topK)
    for iter in 1:(ceil(Int64, log10(N))) ####Changed this line to +1
        for link in ulinks
            p = link.first;q = link.second;
            pmap = munext[p]
            qmap = munext[q]
            if !haskey(pmap, maxmu[q])
                pmap[maxmu[q]] = get(pmap, maxmu[q], 0)
            end
            if !haskey(qmap, maxmu[p])
                qmap[maxmu[p]] = get(qmap, maxmu[p], 0)
            end
            pmap[maxmu[q]] +=  1
            qmap[maxmu[p]] +=  1
        end
        for i in 1:N
            m = munext[i]
            sz = 0
            if length(m) != 0
                if length(m) > topK
                    sz = length(m)
                else
                    sz = topK
                end
                v = [KeyVal(0,0.0) for z in 1:sz]
                c = 1
                for j in m
                    v[c].first = j.first
                    v[c].second = j.second
                    c += 1
                end
                while c <= topK #assign random communities to rest

                    k_= 0

                    while true
                        k_ = sample(1:N)
                        k_ in keys(m) || break
                    end
                    v[c].first = k_
                    v[c].second = _α
                    c+=1
                end
                v = sort_by_values(v)
                mu[i,:]
                for k in 1:topK
                    mu[i,k].first = v[k].first
                    mu[i,k].second = v[k].second + _α
                end
                maxmu[i] = v[1].first
                munext[i] = Dict{Int64, Int64}()
            end
        end
    end
    theta_est = estimate_community_thetas_init(mu,N,topK)
    comms = log_community_groups_init(communities, theta_est, topK, ulinks)
    return comms
end
#need better , change the function below
function random_community_init(N::Int64, num_K::Int64)
	communities = Dict{Int64, Vector{Int64}}()
	for k in 1:(num_K)
		if !haskey(communities, k)
			communities[k] = getkey(communities, k, Int64[])
		end
	end
	partitionsize = div(N, num_K)
	shnodes = shuffle(1:N)
	cur_idx = 1
	cur = shnodes[cur_idx]
	for k in 1:num_K
		count_ = 0
		while count_ != partitionsize
			cur = shnodes[cur_idx]
			push!(communities[k], cur)
			if cur_idx == N
				break
			end
			cur_idx += 1
			count_ += 1
		end
	end
	return communities
end
function init_communities(network::Network{Int64}, N::Int64, findk::Bool, num_K::Int64)
    if findk
    	return batch_community_infer_init(network, N) #this should inside call random
	else
		return random_community_init(N, num_K)
	end
end

function isalink(mg::MetaGraph, a::Int64, b::Int64)
	return has_edge(mg, a, b)
end
#use the lightgraph equivalent instead outneighbors(mg, a)
function neighbors_(mg::MetaGraph, a::Int64)
    return all_neighbors(mg, a)
end
###speed depends on the length of negative_index, memory good overall
#something to think about changes full indices
function negateIndex(full_indices::Vector{Int64}, negative_index::Vector{Int64})
	filter!(i->!in(i,negative_index), full_indices)#removes actual numbers seen
end

# I don't want active candid comms here,
function get_sets_for_init_mu(communities::Dict{Int64, Array{Int64,1}}, mg_t::MetaGraph, N::Int64)
    for k in 1:length(communities)
        communities[k] = unique(communities[k])
    end
    active_comms = VectorList{Int64}()
    for a in 1:N
        push!(active_comms, Int64[])
    end
    for k in collect(keys(communities))
        for a in communities[k]
            active_comms[a] = vcat(active_comms[a],k)
        end
    end
    for a in 1:N
        active_comms[a] = unique(active_comms[a])
        if isempty(active_comms[a])
            active_comms[a] = [ceil(Int64, rand()*length(communities))]
        end
    end
    candid_comms = VectorList{Int64}()
    for a in 1:N
        push!(candid_comms, Int64[])
    end
    for k in 1:length(communities)
        for a in communities[k]
            neighs = unique(neighbors_(mg_t, a))
            for b in neighs
                candid_comms[a] = vcat(candid_comms[a], active_comms[b])
            end
        end
    end
    u = collect(1:length(communities))
    to_edit = Vector{Dict{Int64,Int64}}()

    for a in 1:N
        to_edit = vcat(to_edit,Dict([(i,count(x->x==i,candid_comms[a])) for i in u]))
    end
    edit_keeps = deepcopy(to_edit)
    for a in 1:N
        # to_remove = Int64[]
        for k in collect(keys(to_edit[a]))
            if to_edit[a][k] == 0
                # push!(to_remove, k)
                delete!(edit_keeps[a], k)
            end
        end
    end
    candid_comms = edit_keeps
    ## candid_comms reads this way:
    #  at index a, we look at the active comms of the friends of a
    #  this is shown in a dictionary count that says, a's friends
    #  [k1->k1count, k2->k2count]
    return active_comms, candid_comms

end
#I don't want communities, I only want the inferred or fed K
#This should be based on only getting K
function init_μ(N::Int64, K_feed::Int64, mg::MetaGraph, communities)

    K = K_feed
    _init_μ = zeros(Float64, (K+1, N))
    active_comms, candid_comms = get_sets_for_init_mu(communities, mg, N)

    K = length(communities)

    _init_μ = zeros(Float64, (K+1, N))
    degs = degree(mg)


	activity_weights = log.(1.0.+degs)./log.(N)

	#
    for a in 1:N
        for k in active_comms[a]
            if activity_weights[a] < 0.1
                _init_μ[k,a] += 1.0*sqrt(activity_weights[a])
            else
                _init_μ[k,a] += 1.0
            end
        end
        if activity_weights[a] < 0.1
            _init_μ[K+1,a] = (1.0-activity_weights[a])^2
        else
            _init_μ[K+1,a] = (1.0-sqrt(activity_weights[a]))
        end

        for k in 1:K
            if !(k in active_comms[a])
                _init_μ[k,a] = 1e-3
            end
        end
    end

    for a in 1:N
        _init_μ[:,a] = _init_μ[:,a]./sum(_init_μ[:,a])
        _init_μ[:,a] = log.(_init_μ[:,a])
    end

    for a in 1:N
        _init_μ[:,a] .-= mean(_init_μ[:,a])
    end
######                    ######
######                    ######
    return _init_μ
end


function setup_holdout(hp::Float64, mg_full::MetaGraph, N::Int64)
    hsize = hp*(ne(mg_full)) #num pairs
    hlsize = hnlsize = convert(Int64,div(hsize,2))
    mg_copy = deepcopy(mg_full)
    count_ = 0
    hl_edges = Edge[]
    while count_ < hlsize
        es = [e for e in edges(mg_copy)]
        ei = ceil(Int64,length(es)*rand())
        e = es[ei]
		if degree(mg_copy, e.src) < 4 && degree(mg_copy, e.dst) < 4
			continue;
		end
		if degree(mg_copy, e.src) >20 && degree(mg_copy, e.dst) > 20
	        rem_edge!(mg_copy, e)
	        if !is_connected(Graph(mg_copy))
	            add_edge!(mg_copy, e)
	        else
	            if e in hl_edges
	                continue;
	            else
	                push!(hl_edges, e)
	                count_ += 1
	            end
	        end
		end
    end
    count_ = 0
    hnl_edges = Edge[]
    while count_ < hnlsize
        rrow = ceil(Int64,N*rand())
        rcol= ceil(Int64,N*rand())
        if has_edge(mg_full,rrow, rcol)
            continue
        else
            e = Edge(rrow, rcol)
            if e in hnl_edges
                continue;
            else
                push!(hnl_edges, e)
                count_ +=1
            end
        end
    end

    println("Holdout is set up")
	hl_table = DataFrame()
	hl_table[:first] =zeros(Int64, length(hl_edges))
	hl_table[:second] =zeros(Int64, length(hl_edges))
	hnl_table = DataFrame()
	hnl_table[:first] =zeros(Int64, length(hnl_edges))
	hnl_table[:second] =zeros(Int64, length(hnl_edges))
	for i in 1:length(hl_edges)
		hl_table[i,:first] = hl_edges[i].src
		hl_table[i,:second] = hl_edges[i].dst
		hnl_table[i,:first] = hnl_edges[i].src
		hnl_table[i,:second] = hnl_edges[i].dst
	end
	hl_table = sort(hl_table, :first)
	hnl_table = sort(hnl_table, :first)
    return hl_edges, hnl_edges, mg_copy, hl_table, hnl_table
end
function correct_traindf(hl_edges, hnl_edges, N)
    outhl = zeros(Float64, N)
    inhl = zeros(Float64, N)
    outhnl = zeros(Float64, N)
    inhnl = zeros(Float64, N)
	houtmap = Dict{Int64, Array{Int64, 1}}()
	hinmap = Dict{Int64, Array{Int64, 1}}()
	houtnmap = Dict{Int64, Array{Int64, 1}}()
	hinnmap = Dict{Int64, Array{Int64, 1}}()
    for h in hl_edges
		if !haskey(houtmap, h.src)
			houtmap[h.src] = getkey(houtmap, h.src, Int64[])
		end
		houtmap[h.src] = vcat(houtmap[h.src], h.dst)
		if !haskey(hinmap, h.dst)
			hinmap[h.dst] = getkey(hinmap, h.dst, Int64[])
		end
		hinmap[h.dst] = vcat(hinmap[h.dst], h.src)
        outhl[h.src] +=1
        inhl[h.dst] +=1
    end
    for hn in hnl_edges
		if !haskey(houtnmap, hn.src)
			houtnmap[hn.src] = getkey(houtnmap, hn.src, Int64[])
		end
		houtnmap[hn.src] = vcat(houtnmap[hn.src], hn.dst)
		if !haskey(hinnmap, hn.dst)
			hinnmap[hn.dst] = getkey(hinnmap, hn.dst, Int64[])
		end
		hinnmap[hn.dst] = vcat(hinnmap[hn.dst], hn.src)
        outhnl[hn.src] +=1
        inhnl[hn.dst] +=1
    end
    return outhl,inhl,outhnl,inhnl, houtmap,hinmap,houtnmap,hinnmap
end

function setup_mb_serial(N::Int64, K::Int64,shuffled::Vector{Int64}, nnode::Int64, i::Int64)
	mbnodes = Int64[]
	divisions = N % mbsize > nnode/2 ? div(N, nnode)+1 : div(N, nnode)

	which_division = i % divisions
    if nnode == N
        mbnodes = shuffled
	elseif which_division != 0
		mbnodes = shuffled[nnode*(which_division-1)+1:nnode*(which_division-1) + (nnode)]
	elseif which_division == 0
		mbnodes = shuffled[nnode*(divisions-1)+1 : end]
	end
	zero1 = spzeros(K+1,N)
	zero2 = spzeros(K+1)
	ϕloutsum = zero1[:,:]
	ϕlinsum = zero1[:,:]
	ϕnloutsum = zero1[:,:]
	ϕnlinsum = zero1[:,:]
	ϕlinoutsum = zero2[:]
	ϕnlinoutsum = zero2[:]
	return mbnodes,ϕloutsum,ϕlinsum,ϕlinoutsum,ϕnlinoutsum,ϕnloutsum,ϕnlinsum
end


function estimate_βs!(b::Matrix2d{Float64})
    return b[1,:]./(b[1,:].+b[2,:])
end
function estimate_θs!(theta::Matrix2d{Float64}, mu::Matrix2d{Float64})
	for a in 1:size(theta,2)
		theta[:,a]=softmax(mu[:,a])
	end
end
function estimate_θs_mb!(theta::Matrix2d{Float64}, mu::Matrix2d{Float64}, mbnodes::Vector{Int64})
	for a in mbnodes
		theta[:,a]=softmax(mu[:,a])
	end
end


function update_𝔼lnβ!(𝔼lnβ::Matrix2d{Float64}, b::Matrix2d{Float64}, K::Int64)
	for k in 1:(K+1)
		𝔼lnβ[1,k] = digamma_(b[1,k]) - (digamma_(b[1,k]+b[2,k]))
		𝔼lnβ[2,k] = digamma_(b[2,k]) - (digamma_(b[1,k]+b[2,k]))
	end
end
