push!(LOAD_PATH, "/home/junteng/Documents/publications/repos/spatial_core_periphery/module");

using StatsBase;
using MAT;
using Dierckx;
using Distances;
using Distributions;
using NearestNeighbors;
using SCP_FMM;

#----------------------------------------------------------------
function Euclidean_CoM2(coord1, coord2, m1=1.0, m2=1.0)
    if (m1 == 0.0 && m2 == 0.0)
        m1 = 1.0;
        m2 = 1.0;
    end

    return (coord1*m1+coord2*m2)/(m1+m2);
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
        # learn model parameters
        #----------------------------
        theta, epsilon = SCP_FMM.model_fit(A, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        #----------------------------

        #----------------------------
        # generate random networks
        #----------------------------
        B = SCP_FMM.model_gen(theta, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        #----------------------------
    else
        error("epsilon > 0 in this example.")
    end

    return A, B, theta, coordinates, epsilon;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
# usage: fit to celegans network with Euclidean kernel and FMM
#----------------------------------------------------------------
A,B,theta,coordinates,epsilon = test_celegans(1.0; ratio=0.00, max_num_step=300, opt_epsilon=true)
#----------------------------------------------------------------
