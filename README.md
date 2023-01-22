# Publication-Split-conservative-model-order-reduction-for-hyperbolic-shallow-water-moment-equations

This code framework can be used to reproduce all numerical results of the paper "Split conservative model order reduction for hyperbolic shallow water moment equations using POD Galerkin and DLRA". The code is written in the programming language Julia (Version 1.8.3).

To run the code open Julia and type `]` to open the package manager. Type `activate .` to install all required packages. To load tikzplotlib you have to run the following commands
``
import Pkg; Pkg.add("Conda")

using Conda

Conda.add("tikzplotlib",channel="conda-forge")
``

Then, in the REPL (the Julia environment) type include("runShock.jl") as well as include("runSmoothWave.jl") to reproduce all images. Note that setting up the snapshot matrix for POD Galerkin consumes a large amount of memory and may cause issues depending on you hardware.
