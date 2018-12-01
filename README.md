# Detecting Core-Periphery Structure in Spatial Networks

### This repository hosts the code and some example data for the following paper:  
[Detecting Core-Periphery Structure in Spatial Networks](https://arxiv.org/abs/1808.06544)  
[Junteng Jia](https://000justin000.github.io/) and [Austin R. Benson](https://www.cs.cornell.edu/~arb/)  
arXiv:1808.06544, 2018.

Our paper introduce a random network model for networks with core-periphery structure, where each vertex is associated with a real-valued `core score` to denote its role as a core. A user can do one of the following two things:  
- Generate a random network by providing the predefined `core score` for each vertex.
- Learn core scores of all the vertices by fitting a network to our model. The user need to provide the `adjacency matrix`, `vertex coordinates`, `metric kernel`, and a `center-of-mass` function under the provided metric kernel.

We have demonstrate in our paper that:
- The learned vertex core scores are useful features for data mining purposes. 
- Our algorithms scales to networks with millions of vertices.

Our code is written in Julia 0.6.

### Model inference
Given vertex coordinates and network topology, our model infer the vertex `core scores` by a maximum likelihood estimation. Here is the code snippet for inferencing the vertex core scores for the celegans dataset.

```julia
theta, epsilon = SCP_FMM.model_fit(A, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
```

**Output**: 
- `theta` is the vertex core scores.
- `epsilon` is a parameter in our model that defines the edge length scale.

**Input**: 
- `A` is the |V| x |V| adjacency matrix of the input network.
- `coords` is a d x |V| matrix with the vertex coordinates.
- `Euclidean_CoM2` is a function that compute the center-of-mass between two vertices.
- `Euclidean()` is the metric kernel.
- `epsilon` is the initial value for the length scale parameter.
- `opt` is a dictionary containing user defined parameters.

### Network generation
Given vertex coordinates and vertex core scores, our model sample an instance of random network. Here is the code snippet for generating a random network for celegans dataset.

```julia
B = SCP_FMM.model_gen(theta, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
```

**Output**:
- `B` is the adjacency matrix of the generated random network.

### Examples
For a more detailed explanation, please (see [examples](/examples)). For instance, the following code snippet reproduces figure 4 (A) in our paper.

```julia
include("examples/celegans_naive.jl");
include("examples/celegans_fmm.jl");
using Plots;

Plots.plot(size=(550,500),-5.0:0.1:1.0,-5.0:0.1:1.0,framestyle=:box,label="ideal",color="red",legend=:topleft);
Plots.scatter!(theta0,theta,label="experiment",color="blue")
```

If you have any questions, please email to [jj585@cornell.edu](mailto:jj585@cornell.edu).
