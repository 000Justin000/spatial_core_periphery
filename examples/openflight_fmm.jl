#-------------------------------------------------------------------
# These examples shows how to fit a network using the fmm
# algorithm, to the full model with a metric kernel.
#
# In the fmm algorithm, the pairwise distance between vertices is
# calculated on the fly and the coordinate of vertices are passed
# in to the fitting function. The fmm algorithm takes two arguments
# that allows user to define and use their own kernel function.
# (1) The ``metric'', in this case ``Haversine(6371e3)''.
# (2) The ``center-of-mass'' function used in building the metric
#     tree, in this case ``Haversine_CoM2''.
#
# When computing the objective function and its gradient, as well
# as sampling a network, the fmm algorithm can also explicitly treat
# contributions from a fraction of vertices with the highest core 
# scores, and that fraction is defined in ``opt["ratio"]''. The
# better way to control accuracy is through the accuracy parameters 
# ``opt["delta_1"] and ``opt["delta_2"]. In general, we recommend
# setting ``opt["ratio"] = 0.0'', ``opt["delta_1"] = 2.0'' and
# ``opt["delta_2"] = 0.2''.
#-------------------------------------------------------------------

push!(LOAD_PATH, "./module");

using StatsBase;
using MAT;
using Dierckx;
using Distances;
using Distributions;
using NearestNeighbors;
using SCP_FMM;

#----------------------------------------------------------------
function Haversine_CoM2(coord1, coord2, m1=1.0, m2=1.0)
    if (m1 == 0.0 && m2 == 0.0)
        m1 = 1.0;
        m2 = 1.0;
    end

    lon1 = coord1[1]/180*pi
    lat1 = coord1[2]/180*pi
    lon2 = coord2[1]/180*pi
    lat2 = coord2[2]/180*pi

    x1, y1, z1 = cos(lat1)*cos(lon1), cos(lat1)*sin(lon1), sin(lat1)
    x2, y2, z2 = cos(lat2)*cos(lon2), cos(lat2)*sin(lon2), sin(lat2)

    x = (x1*m1+x2*m2)/(m1+m2)
    y = (y1*m1+y2*m2)/(m1+m2)
    z = (z1*m1+z2*m2)/(m1+m2)

    lon = atan2(y,x)
    hyp = sqrt(x*x+y*y)
    lat = atan2(z,hyp)

    return [lon/pi*180, lat/pi*180]
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_openflight(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true)
    #--------------------------------
    # load airport data and location
    #--------------------------------
    airports_dat = readcsv("./data/open_airlines/airports.dat");
    num_airports = size(airports_dat,1);
    no2id = Dict{Int64, Int64}();
    id2no = Dict{Int64, Int64}();
    id2lc = Dict{Int64, Array{Float64,1}}();
    for i in 1:num_airports
        no2id[i] = airports_dat[i,1];
        id2no[airports_dat[i,1]] = i;
        id2lc[airports_dat[i,1]] = airports_dat[i,7:8];
    end
    #--------------------------------

    #--------------------------------
    W = spzeros(num_airports,num_airports);
    #--------------------------------
    # the adjacency matrix
    #--------------------------------
    routes_dat = readcsv("./data/open_airlines/routes.dat");
    num_routes = size(routes_dat,1);
    for i in 1:num_routes
        id1 = routes_dat[i,4];
        id2 = routes_dat[i,6];
        if (typeof(id1) == Int64 && typeof(id2) == Int64 && haskey(id2lc,id1) && haskey(id2lc,id2))
            W[id2no[id1], id2no[id2]] += 1;
        end
    end
    #--------------------------------
    W = W + W';
    #--------------------------------
    A = spones(sparse(W));
    #--------------------------------

    #--------------------------------
    coordinates = [];
    coords = zeros(2,num_airports);
    for i in 1:num_airports
        push!(coordinates, id2lc[no2id[i]])
        coords[:,i] = flipdim(id2lc[no2id[i]],1)
    end
    #--------------------------------

    opt = Dict();
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = 2.0;
    opt["delta_2"] = 0.2;

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
        theta, epsilon = SCP_FMM.model_fit(A, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        #----------------------------

        #----------------------------
        # generate random networks
        #----------------------------
        B = SCP_FMM.model_gen(theta, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        #----------------------------
    else
        error("epsilon > 0 in this example.")
    end

    return A, B, theta, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
# usage: fit to OpenFlights network with Haversine kernel and FMM
#----------------------------------------------------------------
A,B,theta,coordinates,epsilon = test_openflight(1.0; ratio=0.00, max_num_step=300, opt_epsilon=true)
#----------------------------------------------------------------
