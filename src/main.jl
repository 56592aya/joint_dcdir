include("loader.jl")
include("funcs.jl")


function main(args)

    s = ArgParseSettings()
    @add_arg_table s begin
        # "--file"            #File to read the network from
        #     arg_type = String
        #     required = true
        "--findk"           #will determine the guess of no. of communities
            help = "a flag"
            action = "store_true"
        "-k"               #number of communities
            help = "number of communities, not required if findk"
            arg_type=Int64
            default=0
        "--mbsize"
            help = "number of nodes in a minibatch"
            arg_type=Int64
            default=256
        "--maxiter"
            help = "maximum number of iterations"
            arg_type=Int64
            default=10000
        "--every"
            help = "eval every number of iterations"
            arg_type=Int64
            default=200
    end
    # #
    parsed_args = ArgParse.parse_args(args,s) ##result is a Dict{String, Any}
    println("Parsed args: ")
    for (k,v) in parsed_args
        println("  $k  =>  $(repr(v))")
    end
    println("before parsing")
    # global file = parsed_args["file"]
    global findk = parsed_args["findk"]
    global num_K = parsed_args["k"]
    global mbsize = parsed_args["mbsize"]
    global maxiter = parsed_args["maxiter"]
    global every = parsed_args["every"]
    global iter = maxiter
    global nnode=mbsize
    global evalevery=every


    const findk = true
    const num_K = 0
    const mbsize = 100
    const maxiter=2000
    const iter = maxiter
    const nnode=mbsize
    const every=10
    const evalevery=every

    data = convert(Matrix{Int64},readdlm("n1000k28.txt"))
    Vset=unique(reshape(data, (size(data,1)*2, 1))[:,1])
    network = Network(length(Vset))
    for r in 1:size(data,1)
        network[data[r,1], data[r,2]] = 1
        network[data[r,2], data[r,1]] = 1
    end
    mg=MetaGraph(length(Vset))
    for r in 1:size(data,1)
        add_edge!(mg, data[r,1], data[r,2])
    end
    @assert adjacency_matrix(mg) == network
    @assert length(connected_components(mg)) == 1
    N = nv(mg)
    
