#-----------------------------------------------------------------
# These examples shows how to fit a network using the naive 
# algorithm, either to the basic model, or to the full model
# with a non-metric kernel. In both cases, the FMM algorithm
# can not be used.
#
# In the naive algorithm, the pairwise distance between vertices
# is calculated upfront and stored (passed) in matrix ``D''.
# (1) For the basic model, ``K_{uv} = \delta_{uv}''.
# (2) For the full model with symmetric rank-distance kernel
#     ``D = rank_distance_matrix(Euclidean_matrix(coordinates))''.
#-----------------------------------------------------------------

push!(LOAD_PATH, pwd() * "/../module");

using StatsBase;
using MAT;
using Dierckx;
using Distances;
using Distributions;
using NearestNeighbors;
using SCP;

#----------------------------------------------------------------
function Euclidean_matrix(coords)
    n = size(coords,1);
    D = zeros(n,n);

    for j in 1:n
        for i in j+1:n
            D[i,j] = euclidean(coords[i], coords[j])
        end
    end

    D = D + D';

    @assert issymmetric(D);

    return D;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function rank_distance_matrix(D)
    @assert issymmetric(D);

    n = size(D,1);

    R = zeros(D);
    for j in 1:n
        order = sortperm(D[:,j]);
        for i in 2:n
            R[order[i],j] = i-1;
        end
    end

    R = min.(R,R');

    return R;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_celegans(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true)
    #--------------------------------
    # load celegans data
    #--------------------------------
    data = MAT.matread("../data/celegans/celegans277.mat");
    coords = data["celegans277positions"]';
    A = spones(convert(SparseMatrixCSC{Float64,Int64}, sparse(data["celegans277matrix"] + data["celegans277matrix"]')));
    #--------------------------------

    coordinates = [[coords[1,i], coords[2,i]] for i in 1:size(coords,2)];

    #---------------------------------
    # optimization parameters
    #---------------------------------
    opt = Dict()
    opt["ratio"] = ratio;                   # fraction of vertices computed explicitly
    opt["thres"] = thres;                   # accuracy for termination in L-BFGS algorithm
    opt["max_num_step"] = max_num_step;     # maximal number of steps in L-BFGS algorithm
    opt["opt_epsilon"] = opt_epsilon;       # whether optimize epsilon alongside with core scores
    opt["delta_1"] = 2.0;                   # delta_1 in the paper
    opt["delta_2"] = 0.2;                   # delta_2 in the paper
    #---------------------------------

    #--------------------------------
    # epsilon: initial value for epsilon, 
    #--------------------------------
    #   (1) set ``epsilon=0'' and ``opt["opt_epsilon"]=false'' is equivalent to the basic model
    #   (2) set ``epsilon>0'' and ``opt["opt_epsilon"]=true'' uses metric kernel
    #   (3) set ``epsilon<0'' and ``opt["opt_epsilon"]=true'' uses rank-distance kernel
    #--------------------------------
    if (epsilon > 0)
        #----------------------------
        error("epsilon <= 0 in these examples.")
        #----------------------------
    elseif (epsilon < 0)
        #----------------------------
        epsilon *= -1.0;

        D = rank_distance_matrix(Euclidean_matrix(coordinates));

        #----------------------------
        # learn model parameters
        #----------------------------
        theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
        #----------------------------

        #----------------------------
        # generate random networks
        #----------------------------
        B = SCP.model_gen(theta, D, epsilon);
        #----------------------------

        epsilon *= -1.0;
        #----------------------------
    else
        #----------------------------
        D = ones(A)-eye(A);

        #----------------------------
        # learn model parameters
        #----------------------------
        theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
        #----------------------------

        #----------------------------
        # generate random networks
        #----------------------------
        B = SCP.model_gen(theta, D, epsilon);
        #----------------------------
    end

    return A, B, theta, coordinates, epsilon;
end



#----------------------------------------------------------------
# usage: fit to celegans network to the basic model
#----------------------------------------------------------------
A,B,theta,coordinates,epsilon = test_celegans(0.0; ratio=1.00, max_num_step=300, opt_epsilon=false)
#----------------------------------------------------------------

#----------------------------------------------------------------
# usage: fit to celegans network with rank distance kernel
#----------------------------------------------------------------
# A,B,theta,coordinates,epsilon = test_celegans(-1.0; ratio=1.00, max_num_step=300, opt_epsilon=true)
#----------------------------------------------------------------
