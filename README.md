# spatial_core_periphery

#### This repository hosts the code and some example data for the following paper:  
[Detecting Core-Periphery Structure in Spatial Networks](https://arxiv.org/abs/1808.06544)  
[Junteng Jia](https://000justin000.github.io/) and [Austin R. Benson](https://www.cs.cornell.edu/~arb/)  
arXiv:1808.06544, 2018.

Our paper introduce a random network model for networks with core-periphery structure, where each vertex is associated with a real-valued `core score` to denote its role as a core. A user can do one of the following two things (see [examples](/examples)):  
- Generate a random network by providing the predefined `core score` for each vertex.
- Learn core scores of all the vertices by fitting a network to our model. The user need to provide the `adjacency matrix`, `vertex coordinates`, `metric kernel`, and a `center-of-mass` function under the provided metric kernel.

We have demonstrate in our paper that:
- The learned vertex core scores are useful features for data mining purposes. 
- Our algorithms scales to networks with millions of vertices.

If you have any questions, please email to [jj585@cornell.edu](mailto:jj585@cornell.edu).