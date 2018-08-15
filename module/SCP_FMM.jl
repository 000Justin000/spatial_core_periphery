#---------------------------------------------------------------------------------
module SCP_FMM
    using StatsBase;
    using Distances;
    using NearestNeighbors;
    using Optim;

    export model_fit, model_gen


    #-----------------------------------------------------------------------------
    # data structure providing information of a given node
    #-----------------------------------------------------------------------------
    mutable struct Particle
        CoM::Vector{Float64}
        m::Float64
        maxm::Float64
        pot_1::Float64
        pot_2::Float64
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    function subtree_size(idx, n)
        @assert mod(n,2) == 1;
        @assert idx <= n;

        p = Int64(floor(log(n)/log(2)) - floor(log(idx)/log(2)));

        if (2^p * idx + 2^p - 1 <= n) # the tree that root at idx is full binary tree with height p
            size = 2^p;
        elseif (2^p * idx <= n)       # the tree that root at idx is (not full) binary tree with height p
            size = 2^(p-1) + div(n - 2^p * idx + 1, 2);
        else                          # the tree that root at idx is full binary tree with height p-1
            size = 2^(p-1);
        end

        return size;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    function subtree_range(idx, n)
        @assert mod(n,2) == 1;
        @assert idx <= n;

        p = Int64(floor(log(n)/log(2)) - floor(log(idx)/log(2)));

        if (2^p * idx + 2^p - 1 <= n) # the tree that root at idx is full binary tree with height p
            range = collect(2^p * idx : 2^p * idx + 2^p - 1);
        elseif (2^p * idx <= n)       # the tree that root at idx is (not full) binary tree with height p
            range = vcat(collect(2^(p-1) * idx + div(n - 2^p * idx + 1, 2) : 2^(p-1) * idx + 2^(p-1) - 1), collect(2^p * idx : n));
        else                          # the tree that root at idx is full binary tree with height p-1
            range = collect(2^(p-1) * idx : 2^(p-1) * idx + 2^(p-1) - 1);
        end

        return range;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively compute the center of mass of each node
    #-----------------------------------------------------------------------------
    function fill_cm!(cmp, idx, bt, ms, roid, srid, CoM2)
        if (idx > bt.tree_data.n_internal_nodes)
            cmp[idx] = Particle(convert(Vector{Float64}, bt.hyper_spheres[idx].center),
                                ms[roid[srid[idx]]], ms[roid[srid[idx]]], 0.0, 0.0);
        else
            if (idx*2+1 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
                fill_cm!(cmp, idx*2,   bt, ms, roid, srid, CoM2);
                fill_cm!(cmp, idx*2+1, bt, ms, roid, srid, CoM2);
                cmp[idx] = Particle(CoM2(cmp[idx*2].CoM,cmp[idx*2+1].CoM, cmp[idx*2].m,cmp[idx*2+1].m),
                                    cmp[idx*2].m+cmp[idx*2+1].m, max(cmp[idx*2].maxm, cmp[idx*2+1].maxm), 0.0, 0.0);
            elseif (idx*2 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
                fill_cm!(cmp, idx*2,   bt, ms, roid, srid, CoM2);
                cmp[idx] = Particle(cmp[idx*2].CoM, cmp[idx*2].m, cmp[idx*2].m, cmp[idx*2].pot_1, cmp[idx*2].pot_2);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the interaction between two nodes
    #-----------------------------------------------------------------------------
    function accumulate_i2!(cmp, idx_1, idx_2, bt, epsilon, opt)
        n_node = bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs;
        if ((idx_1 <= n_node) && (idx_2 <= n_node))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center);
            #-----------------------------------------------------------------
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            #-----------------------------------------------------------------
            if ((idx_1 > bt.tree_data.n_internal_nodes) && (idx_2 > bt.tree_data.n_internal_nodes))
                cmp[end].pot_1 += log(1 + (cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon));
                cmp[end].m += 1;
            elseif (distance >= max(epsilon*2, opt["delta_1"])*(sp1r + sp2r) && ((cmp[idx_1].maxm * cmp[idx_2].maxm)/(distance^epsilon) < opt["delta_2"]))
            # elseif ((sp1r + sp2r) < 1.0e-12)
                cmp[end].pot_1 += +(1/1) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^1
                                  -(1/2) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^2
                                  +(1/3) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^3
                                  -(1/4) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^4;
                cmp[end].m += 1;
            elseif (sp1r <= sp2r)
                accumulate_i2!(cmp, idx_1, idx_2*2,   bt, epsilon, opt);
                accumulate_i2!(cmp, idx_1, idx_2*2+1, bt, epsilon, opt);
            else
                accumulate_i2!(cmp, idx_1*2,   idx_2, bt, epsilon, opt);
                accumulate_i2!(cmp, idx_1*2+1, idx_2, bt, epsilon, opt);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively compute the interaction at each level of the tree
    #-----------------------------------------------------------------------------
    function accumulate_i!(cmp, idx, bt, epsilon, opt)
        if (idx <= bt.tree_data.n_internal_nodes)
            accumulate_i2!(cmp, idx*2, idx*2+1, bt, epsilon, opt)
            accumulate_i!(cmp,  idx*2,          bt, epsilon, opt)
            accumulate_i!(cmp,  idx*2+1,        bt, epsilon, opt)
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the expected degree of each node with fast multipole method
    #-----------------------------------------------------------------------------
    function omega!(theta::Array{Float64,1}, coords, CoM2, dist, epsilon, bt, A, sum_logD_inE, opt)
        # println(" - da"); # debug

        n = length(theta);

        #-------------------------------------------------------------------------
        omega = 0.0;
        #-------------------------------------------------------------------------
        I,J,V = findnz(A);
        #-----------------------------------------------------------------------------
        for (i,j) in zip(I,J)
            #---------------------------------------------------------------------
            if (i < j)
                omega += theta[i] + theta[j];
            end
            #---------------------------------------------------------------------
        end
        #-----------------------------------------------------------------------------
        omega -= epsilon * sum_logD_inE;
        #-----------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        core_id = sortperm(theta, rev=true)[1:Int64(ceil(opt["ratio"] * n))];
        #-------------------------------------------------------------------------
        # now compute the c->c and c->p
        #-------------------------------------------------------------------------
        for cid in core_id
            if (!haskey(dist, cid))
                dist2cid = zeros(n);
                for i in 1:n
                    dist2cid[i] = evaluate(bt.metric, coords[:,cid], coords[:,i]);
                end

                dist[cid] = dist2cid;
            end

            #---------------------------------------------------------------------
            cid2all = log.( (exp.(theta[cid] .+ theta) .+ dist[cid].^epsilon) ./ (dist[cid].^epsilon) );
            cid2all[cid] = 0;     omega -= 0.5 * sum(cid2all);     # c->c and c->p
            cid2all[core_id] = 0; omega -= 0.5 * sum(cid2all);              # c->p
            #---------------------------------------------------------------------
        end
        #-------------------------------------------------------------------------

#        println("c->c, c->p finished");
#        @assert !isnan(omega);

        #-------------------------------------------------------------------------
        # now compute the p->c and p->p
        #-------------------------------------------------------------------------
        td = bt.tree_data;
        ni = td.n_internal_nodes;
        nl = td.n_leafs;
        roid = bt.indices;             # (o)riginal id of the (r)eordered data points
        orid = sortperm(roid);         # (r)eordered id of the (o)riginal data points
        #-------------------------------------------------------------------------
        # (r)eordered data point id of the hyper(s)pheres
        srid = Dict((idx >= td.cross_node) ? (idx => td.offset_cross + idx) : (idx => td.offset + idx) for idx in (ni+1:ni+nl));
        # hyper(s)phere id of the (r)eordered data points
        rsid = Dict((idx >= td.cross_node) ? (td.offset_cross + idx => idx) : (td.offset + idx => idx) for idx in (ni+1:ni+nl));
        #-------------------------------------------------------------------------
        # data structure that stores the node's CoM, mass, and potential
        # corresponding to nodes in BallTree data structure
        #-------------------------------------------------------------------------
        ms = exp.(theta); ms[core_id] = 0;
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        fmm_tree = Array{Particle,1}(ni+nl+1);
        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0, 0.0);
        #-------------------------------------------------------------------------
        fill_cm!(fmm_tree, 1, bt, ms, roid, srid, CoM2);
        accumulate_i!(fmm_tree, 1, bt, epsilon, opt);
        #-------------------------------------------------------------------------
        omega -= fmm_tree[end].pot_1;
        #-------------------------------------------------------------------------

        return omega;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the potential between two nodes
    #-----------------------------------------------------------------------------
    function fill_p2!(cmp, idx_1, idx_2, bt, epsilon, opt)
        n_node = bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs;
        if ((idx_1 <= n_node) && (idx_2 <= n_node))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center);
            #-----------------------------------------------------------------
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            #-----------------------------------------------------------------
            if ((idx_1 > bt.tree_data.n_internal_nodes) && (idx_2 > bt.tree_data.n_internal_nodes))
                cmp[idx_1].pot_1 += cmp[idx_2].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon);
                cmp[idx_2].pot_1 += cmp[idx_1].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon);
                cmp[idx_1].pot_2 += cmp[idx_2].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon) * log(distance);
                cmp[idx_2].pot_2 += cmp[idx_1].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon) * log(distance);
                cmp[end].m += 1;
            elseif (distance >= max(epsilon*2, opt["delta_1"])*(sp1r + sp2r) && ((cmp[idx_1].maxm * cmp[idx_2].maxm)/(distance^epsilon) < opt["delta_2"]))
            # elseif ((sp1r + sp2r) < 1.0e-12)
                cmp[idx_1].pot_1 += cmp[idx_2].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon);
                cmp[idx_2].pot_1 += cmp[idx_1].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon);
                cmp[idx_1].pot_2 += cmp[idx_2].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon) * log(distance);
                cmp[idx_2].pot_2 += cmp[idx_1].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon) * log(distance);
                cmp[end].m += 1;
            elseif (sp1r <= sp2r)
                fill_p2!(cmp, idx_1, idx_2*2,   bt, epsilon, opt);
                fill_p2!(cmp, idx_1, idx_2*2+1, bt, epsilon, opt);
            else
                fill_p2!(cmp, idx_1*2,   idx_2, bt, epsilon, opt);
                fill_p2!(cmp, idx_1*2+1, idx_2, bt, epsilon, opt);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively compute the potential at each level of the tree
    #-----------------------------------------------------------------------------
    function fill_p!(cmp, idx, bt, epsilon, opt)
        if (idx <= bt.tree_data.n_internal_nodes)
            fill_p2!(cmp, idx*2, idx*2+1, bt, epsilon, opt)
            fill_p!(cmp,  idx*2,          bt, epsilon, opt)
            fill_p!(cmp,  idx*2+1,        bt, epsilon, opt)
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively accumulate the potential to the lowest level
    #-----------------------------------------------------------------------------
    function accumulate_p!(cmp, idx, bt)
        if (2 <= idx <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
            cmp[idx].pot_1 += cmp[div(idx,2)].pot_1
            cmp[idx].pot_2 += cmp[div(idx,2)].pot_2
        end

        if (idx <= bt.tree_data.n_internal_nodes)
            accumulate_p!(cmp, idx*2,   bt)
            accumulate_p!(cmp, idx*2+1, bt)
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the expected degree of each node with fast multipole method
    #-----------------------------------------------------------------------------
    function epd_and_srd!(theta::Array{Float64,1}, coords, CoM2, dist, epsilon, bt, opt)
        n = length(theta);

        #-------------------------------------------------------------------------
        # "expected degree" and "sum_rho_logD"
        #-------------------------------------------------------------------------
        epd = zeros(n);
        srd = 0.0;
        #-------------------------------------------------------------------------
        core_id = sortperm(theta, rev=true)[1:Int64(ceil(opt["ratio"] * n))];
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # compute the expected degree and sum_rho_logD exactly for core nodes
        #-------------------------------------------------------------------------
        epdc = Dict{Int64, Float64}();
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # now compute the c->c and c->p
        #-------------------------------------------------------------------------
        for cid in core_id
            if (!haskey(dist, cid))
                dist2cid = zeros(n);
                for i in 1:n
                    dist2cid[i] = evaluate(bt.metric, coords[:,cid], coords[:,i]);
                end

                dist[cid] = dist2cid;
            end

            #---------------------------------------------------------------------
            cid2all_1 = exp.(theta[cid] .+ theta) ./ (exp.(theta[cid] .+ theta) .+ dist[cid].^epsilon);
            cid2all_1[cid] = 0; epd += cid2all_1;
            epdc[cid] = sum(cid2all_1);                      # store c->c and c->p
            #---------------------------------------------------------------------
            cid2all_2 = exp.(theta[cid] .+ theta) ./ (exp.(theta[cid] .+ theta) .+ dist[cid].^epsilon) .* log.(dist[cid]);
            cid2all_2[cid] = 0;     srd += sum(cid2all_2);         # c->c and c->p
            cid2all_2[core_id] = 0; srd += sum(cid2all_2);         # p->c
            #---------------------------------------------------------------------
        end
        #-------------------------------------------------------------------------

#       println("c->c, c->p finished!");

        #-------------------------------------------------------------------------
        # now compute the p->c and p->p
        #-------------------------------------------------------------------------
        td = bt.tree_data;
        ni = td.n_internal_nodes;
        nl = td.n_leafs;
        roid = bt.indices;             # (o)riginal id of the (r)eordered data points
        orid = sortperm(roid);         # (r)eordered id of the (o)riginal data points
        #-------------------------------------------------------------------------
        # (r)eordered data point id of the hyper(s)pheres
        srid = Dict((idx >= td.cross_node) ? (idx => td.offset_cross + idx) : (idx => td.offset + idx) for idx in (ni+1:ni+nl));
        # hyper(s)phere id of the (r)eordered data points
        rsid = Dict((idx >= td.cross_node) ? (td.offset_cross + idx => idx) : (td.offset + idx => idx) for idx in (ni+1:ni+nl));
        #-------------------------------------------------------------------------
        # data structure that stores the node's CoM, mass, and potential
        # corresponding to nodes in BallTree data structure
        #-------------------------------------------------------------------------
        ms = exp.(theta); ms[core_id] = 0;
        #-------------------------------------------------------------------------
        fmm_tree = Array{Particle,1}(ni+nl+1);
        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0, 0.0);
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        fill_cm!(fmm_tree, 1, bt, ms, roid, srid, CoM2);
        fill_p!(fmm_tree, 1, bt, epsilon, opt);
        accumulate_p!(fmm_tree, 1, bt);
        #-------------------------------------------------------------------------
        epd += [fmm_tree[rsid[orid[idx]]].pot_1 for idx in 1:nl] .* ms;
        #-------------------------------------------------------------------------
        srd += sum([fmm_tree[rsid[orid[idx]]].pot_2 for idx in 1:nl] .* ms);
        #-------------------------------------------------------------------------
        srd /= 2.0;
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # replace with the (stored) exact expected degree for core nodes
        #-------------------------------------------------------------------------
        for cid in core_id
            epd[cid] = epdc[cid];
        end
        #-------------------------------------------------------------------------

        return epd, srd, fmm_tree;
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # gradient of objective function
    #-----------------------------------------------------------------------------
    function negative_gradient_omega!(theta, coords, CoM2, dist, epsilon, bt, d, sum_logD_inE, storage, opt)
        # print(maximum(theta), "    ", minimum(theta), "    ", mean(theta), "    di"); # debug

        epd, srd, fmm_tree = epd_and_srd!(theta, coords, CoM2, dist, epsilon, bt, opt);

        G = d - epd;

        storage[1:end-1] = -G;
        storage[end] = opt["opt_epsilon"] ? -(srd - sum_logD_inE) : 0.0;
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # given the adjacency matrix and coordinates, compute the core scores
    #-----------------------------------------------------------------------------
    function model_fit(A::SparseMatrixCSC{Float64,Int64},
                       coords::Array{Float64,2},
                       CoM2,
                       metric = Euclidean(),
                       epsilon = 1;
                       opt = Dict("thres"=>1.0e-6, "max_num_step"=>10000, "ratio"=>1.0, "opt_epsilon"=>true, "delta_1"=>2.0, "delta_2"=>0.2),
                       theta0 = nothing)
        @assert issymmetric(A);

        #-----------------------------------------------------------------------------
        A = spones(A);
        n = size(A,1);
        d = vec(sum(A,2));
        order = sortperm(d, rev=true);
        #-----------------------------------------------------------------------------
        if (theta0 == nothing)
            theta = d / maximum(d) * 1.0e-6;
        else
            theta = theta0;
        end
        #-----------------------------------------------------------------------------

        #-----------------------------------------------------------------------------
        # \sum_{ij in E} -log_Dij
        #-----------------------------------------------------------------------------
        I,J,V = findnz(A);
        #-----------------------------------------------------------------------------
        sum_logD_inE = 0.0;
        #-----------------------------------------------------------------------------
        for (i,j) in zip(I,J)
            #---------------------------------------------------------------------
            if (i < j)
                sum_logD_inE += log(evaluate(metric, coords[:,i], coords[:,j]));
            end
            #---------------------------------------------------------------------
        end
        #-----------------------------------------------------------------------------
        # println("sum_logD_inE: ", sum_logD_inE);

        dist = Dict{Int64,Array{Float64,1}}();
        bt = BallTree(coords, metric, leafsize=1);

        f!(x)          = -omega!(x[1:end-1], coords, CoM2, dist, x[end], bt, A, sum_logD_inE, opt);
        g!(storage, x) =  negative_gradient_omega!(x[1:end-1], coords, CoM2, dist, x[end], bt, d, sum_logD_inE, storage, opt)

        #-----------------------------------------------------------------------------
        println("starting optimization:")
        #-----------------------------------------------------------------------------
        precond = speye(length(theta)+1) * length(theta); precond[end,end] *= length(theta);
        optim = optimize(f!, g!, vcat(theta,[epsilon]), LBFGS(P = precond), Optim.Options(g_tol = 1e-6,
                                                                                      iterations = opt["max_num_step"],
                                                                                      show_trace = true,
                                                                                      show_every = 1,
                                                                                      allow_f_increases = false));
        #-----------------------------------------------------------------------------
        println(optim);
        #-----------------------------------------------------------------------------

        theta = optim.minimizer[1:end-1];
        epsilon = optim.minimizer[end];

        println(epsilon);
        println(omega!(theta, coords, CoM2, dist, epsilon, bt, A, sum_logD_inE, opt));

        @assert epsilon > 0;
        return theta, epsilon;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # generate edges between points within two hyper_spheres
    #-----------------------------------------------------------------------------
    function generate_e2!(cmp, idx_1, idx_2, bt, epsilon, roid, srid, I,J,V, opt)
        n_node = bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs;
        if ((idx_1 <= n_node) && (idx_2 <= n_node))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center);
            #-----------------------------------------------------------------
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            #-----------------------------------------------------------------
            if ((idx_1 > bt.tree_data.n_internal_nodes) && (idx_2 > bt.tree_data.n_internal_nodes))
                if (rand() < (cmp[idx_1].m * cmp[idx_2].m)/((cmp[idx_1].m * cmp[idx_2].m) + distance^epsilon))
                    push!(I,roid[srid[idx_1]]);
                    push!(J,roid[srid[idx_2]]);
                    push!(V,1.0);
                end

                cmp[end].m += 1;
            elseif (distance >= max(epsilon*2, opt["delta_1"])*(sp1r + sp2r) && ((cmp[idx_1].maxm * cmp[idx_2].maxm)/(distance^epsilon) < opt["delta_2"]))
            # elseif ((sp1r + sp2r) < 1.0e-12)
                nef = (cmp[idx_1].m * cmp[idx_2].m) / ((cmp[idx_1].m * cmp[idx_2].m)/(subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon);
                nei = Int64(floor(nef) + (rand() < (nef - floor(nef)) ? 1 : 0));

                # generate nei edges between this two group of nodes
                grp_1 = subtree_range(idx_1,n_node);
                grp_2 = subtree_range(idx_2,n_node);

                grp_1_mass = [cmp[it].m for it in grp_1];
                grp_2_mass = [cmp[it].m for it in grp_2];

                grp_1_mass_mean = mean(grp_1_mass);
                grp_2_mass_mean = mean(grp_2_mass);

                grp_1_prob = (grp_1_mass .* grp_2_mass_mean) ./ (grp_1_mass .* grp_2_mass_mean + distance^epsilon); grp_1_prob /= sum(grp_1_prob);
                grp_2_prob = (grp_2_mass .* grp_1_mass_mean) ./ (grp_2_mass .* grp_1_mass_mean + distance^epsilon); grp_2_prob /= sum(grp_2_prob);

                generated_edges = Set{Tuple{Int64,Int64}}();

                #--------------------------------------------------
                # ball dropping
                #--------------------------------------------------
                grp_1_ids = StatsBase.sample(grp_1, Weights(grp_1_prob), 2*nei, replace=true); # sample nei nodes from grp_1 with grp_1_prob
                grp_2_ids = StatsBase.sample(grp_2, Weights(grp_2_prob), 2*nei, replace=true); # sample nei nodes from grp_2 with grp_2_prob

                for i in 1:2*nei
                    oid_1 = roid[srid[grp_1_ids[i]]];
                    oid_2 = roid[srid[grp_2_ids[i]]];
                    current_edge = (min(oid_1,oid_2), max(oid_1,oid_2));

                    if (!(current_edge in generated_edges))
                        push!(generated_edges, current_edge);
                        #------------------------------------
                        push!(I,current_edge[1]);
                        push!(J,current_edge[2]);
                        push!(V,1.0);
                    end

                    if (length(generated_edges) >= nei)
                        break;
                    end
                end
                #--------------------------------------------------

#               #--------------------------------------------------
#               # grass hopping
#               #--------------------------------------------------
#               grp_1_bin = vcat([0], cumsum(grp_1_prob)[1:end-1]);
#               grp_2_bin = vcat([0], cumsum(grp_2_prob)[1:end-1]);
#               #--------------------------------------------------
#               offset = rand() * 1.0/nei;
#               #--------------------------------------------------
#               for i in 0:nei-1
#                   target = i/nei + offset;
#                   #----------------------------------------------
#                   gid_1 = searchsortedlast(grp_1_bin, target);
#                   gid_2 = searchsortedlast(grp_2_bin*grp_1_prob[gid_1], target-grp_1_bin[gid_1]);
#                   #----------------------------------------------
#                   oid_1 = roid[srid[grp_1[gid_1]]];
#                   oid_2 = roid[srid[grp_2[gid_2]]];
#                   current_edge = (min(oid_1,oid_2), max(oid_1,oid_2));
#                   #----------------------------------------------
#                   @assert !(current_edge in generated_edges);
#                   push!(generated_edges, current_edge);
#                   #----------------------------------------------
#                   push!(I,current_edge[1]);
#                   push!(J,current_edge[2]);
#                   push!(V,1.0);
#               end
#               #--------------------------------------------------

                cmp[end].m += 1;
            elseif (sp1r <= sp2r)
                generate_e2!(cmp, idx_1, idx_2*2,   bt, epsilon, roid, srid, I,J,V, opt);
                generate_e2!(cmp, idx_1, idx_2*2+1, bt, epsilon, roid, srid, I,J,V, opt);
            else
                generate_e2!(cmp, idx_1*2,   idx_2, bt, epsilon, roid, srid, I,J,V, opt);
                generate_e2!(cmp, idx_1*2+1, idx_2, bt, epsilon, roid, srid, I,J,V, opt);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively generate edges at each level of the tree
    #-----------------------------------------------------------------------------
    function generate_e!(cmp, idx, bt, epsilon, roid, srid, I,J,V, opt)
        if (idx <= bt.tree_data.n_internal_nodes)
            generate_e2!(cmp, idx*2, idx*2+1, bt, epsilon, roid, srid, I,J,V, opt);
            generate_e!(cmp,  idx*2,          bt, epsilon, roid, srid, I,J,V, opt);
            generate_e!(cmp,  idx*2+1,        bt, epsilon, roid, srid, I,J,V, opt);
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # given the core score and distance matrix, compute the adjacency matrix
    #-----------------------------------------------------------------------------
    function model_gen(theta::Array{Float64,1},
                       coords::Array{Float64,2},
                       CoM2,
                       metric = Euclidean(),
                       epsilon = 1;
                       opt = Dict("ratio"=>1.0, "delta_1"=>2.0, "delta_2"=>0.2))

        I = Vector{Int64}();
        J = Vector{Int64}();
        V = Vector{Float64}();

        n = length(theta);

        dist = Dict{Int64,Array{Float64,1}}();
        bt = BallTree(coords, metric, leafsize=1);

        #-------------------------------------------------------------------------
        core_id  = sortperm(theta, rev=true)[1:Int64(ceil(opt["ratio"] * n))];
        core_set = Set(core_id);
        #-------------------------------------------------------------------------
        # now compute the c->c and c->p
        #-------------------------------------------------------------------------
        # println(length(core_id));
        #-------------------------------------------------------------------------
        for cid in core_id
            if (!haskey(dist, cid))
                dist2cid = zeros(n);
                for i in 1:n
                    dist2cid[i] = evaluate(bt.metric, coords[:,cid], coords[:,i]);
                end

                dist[cid] = dist2cid;
            end

            for i in 1:n
                if (!(i in core_set && i <= cid))
                    if (rand() < exp(theta[cid]+theta[i])/(exp(theta[cid]+theta[i]) + dist[cid][i]^epsilon))
                        push!(I, cid);
                        push!(J, i);
                        push!(V, 1.0);
                    end
                end
            end
        end
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # now compute the p->c and p->p
        #-------------------------------------------------------------------------
        td = bt.tree_data;
        ni = td.n_internal_nodes;
        nl = td.n_leafs;
        roid = bt.indices;             # (o)riginal id of the (r)eordered data points
        orid = sortperm(roid);         # (r)eordered id of the (o)riginal data points
        #-------------------------------------------------------------------------
        # (r)eordered data point id of the hyper(s)pheres
        srid = Dict((idx >= td.cross_node) ? (idx => td.offset_cross + idx) : (idx => td.offset + idx) for idx in (ni+1:ni+nl));
        # hyper(s)phere id of the (r)eordered data points
        rsid = Dict((idx >= td.cross_node) ? (td.offset_cross + idx => idx) : (td.offset + idx => idx) for idx in (ni+1:ni+nl));
        #-------------------------------------------------------------------------
        # data structure that stores the node's CoM, mass, and potential
        # corresponding to nodes in BallTree data structure
        #-------------------------------------------------------------------------
        ms = exp.(theta); ms[core_id] = 0;
        #-------------------------------------------------------------------------
        fmm_tree = Array{Particle,1}(ni+nl+1);
        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0, 0.0);
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        fill_cm!(fmm_tree, 1, bt, ms, roid, srid, CoM2);
        generate_e!(fmm_tree, 1, bt, epsilon, roid, srid, I,J,V, opt);
        #-------------------------------------------------------------------------
        # println(sum(A));
        #-------------------------------------------------------------------------

        A = sparse(I,J,V, n,n,max);

        return A + A';
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
