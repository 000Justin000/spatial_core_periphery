#-------------------------------------------------------------------
# These examples performs benchmark scaling test of our algorithm.
#
# There are two major functions in this file,
# (1) ``synthetic_network'' takes parameters in our model as inputs
#     returns the adjacency matrix of a random network
# (2) ``''
#-------------------------------------------------------------------

push!(LOAD_PATH, "./module");

using StatsBase;
using Distances;
using NearestNeighbors;
using SCP;
using SCP_FMM;

#----------------------------------------------------------------
function Euclidean_CoM2(coordinate1, coordinate2, m1=1.0, m2=1.0)
    if (m1 == 0.0 && m2 == 0.0)
        m1 = 1.0;
        m2 = 1.0;
    end

    return (coordinate1*m1+coordinate2*m2)/(m1+m2);
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function Euclidean_matrix(coordinates)
    n = size(coordinates,1);
    D = zeros(n,n);

    for j in 1:n
        for i in j+1:n
            D[i,j] = euclidean(coordinates[i], coordinates[j])
        end
    end

    D = D + D';

    @assert issymmetric(D);

    return D;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function synthetic_network(n::Int64=100, epsilon=2.0; metric=Euclidean(), CoM2=Euclidean_CoM2, opt=nothing)
    #------------------------------------------------------------
    coordinates = rand(2,n);
    #------------------------------------------------------------
    bt = BallTree(coordinates, metric, leafsize=1);
    #------------------------------------------------------------

    #------------------------------------------------------------
    TT = Dict(100 => -2.25, 1000 => -3.77, 10000 => -5.13, 100000 => -6.43, 1000000 => -7.69);
    @assert n in keys(TT);
    #------------------------------------------------------------
    
    #------------------------------------------------------------
    theta = ones(n) * TT[n];
    #------------------------------------------------------------
    theta[1:Int64(ceil(0.05*n))] += 1.0;
    #------------------------------------------------------------

    A = SCP_FMM.model_gen(theta, coordinates, CoM2, metric, epsilon; opt = (opt == nothing ? Dict("ratio"=>0.0, "delta_1" => 2.0, "delta_2" => 0.2) : opt));

    return coordinates, theta, A;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function benchmark_synthetic_network(n, epsilon0; ratio=0.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true, delta_1=2.0, delta_2=0.2, use_fmm=false)
    #--------------------------------
    # n : number of vertices in the random network
    #--------------------------------
    # set parameters in our model and
    # generate a random network
    #--------------------------------
    opt = Dict();
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = delta_1;
    opt["delta_2"] = delta_2;
    #--------------------------------
    coordinates, theta0, A = synthetic_network(n, epsilon0; opt=opt);
    #--------------------------------
    
    #--------------------------------
    # infer model parameters
    #--------------------------------
    if (use_fmm)
        theta, epsilon, optim = SCP_FMM.model_fit(A, coordinates, Euclidean_CoM2, Euclidean(), 1.0; opt=opt);
    else
        D = Euclidean_matrix([coordinates[:,i] for i in 1:size(coordinates,2)]);
        theta, epsilon, optim = SCP.model_fit(A, D, 1.0; opt=opt);
    end
    #--------------------------------
    
    println("#########################################");
    println("error in epsilon: ", epsilon-epsilon0);
    println("#########################################");

    return optim;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
optim = benchmark_synthetic_network(100, 2.0; max_num_step=100, use_fmm=false);
#----------------------------------------------------------------
# optim = benchmark_synthetic_network(1000, 2.5; max_num_step=100, use_fmm=true);
#----------------------------------------------------------------
