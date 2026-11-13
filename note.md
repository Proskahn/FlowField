For the src/problems, what should be included in there and what should not?

1. The problem should be defined, including the geometry and physics. In short, it should include the simulation problem.

2. However, the problem means optimization problem, which means the variables and objectives should also be defined within the problems. Of course, it should be written by some python code within the folder. But the code should 'extract' the information from the openFOAM simulation result and make the CFD simulation independent.

3. I am wondering, should the src/problems be limited on the above scope. It means, when an external file inquiry the variable by giving a vector, it feeds the objective back.


## Optimization Problem
Since all flowfield simulation problems have some common interface, it's smart to have a abstract class with name like 'AbstractFlowFieldClass' where the following methods could be included.
(1). Validate the variables
(2). Access the geometry
(3). write case files and run the simulation.
(4). Define the objectives
Wow you already have it in the src/flowfield/base.py, I didn't notice that before. Maybe give it a better class name and file name to make it easier to find?

## Mesh
Regarding the geometry, the mesh generation of each problem should also share something. At this moment, class method is not applied. Is class method better here? The question is, people told me, use function method if class method is not necessary.

What is the output of the Mesh generation? Polymesh

An abstract class with potential name like AbstractMeshGeneration might also be interesting.


## Evaluation
Different problems have similar run_case and evaluate function, is it possible to unify it? For example, define a public evaluator class, while read the problem-specific configuration from other documents.

## Config
For the config, is it necessary to make it a class? There is some validation functions but just validates the string input. Is it very necessary? Does json also work here, when we put the validation in another place? I am not sure, it's up to your suggestion.

## Optimization
To run the optimization, 

(1). The optimizaiton should be in a seperate folder instead of staying with the problem.

(2). To run the optimization, I hope that a specific problem name and optimizer can be specified. This is because I am always adding problems and optimizers. To avoid run the entire benchmark, it might be more convenient to seperate those problems and store the running data for later plot.

(3). I am very sure that several different Bayesian Optimization algorithms and metaheuristics algorithms, will be iplemented, such as TurBO, Vanilla BO, NSGA-II, CMA-ES. In this case, do you think it might be more convenient to have an Sampler class?

## Benchmarking

For the benchmarking, only performance comparison of single optimizer might not be enough, the IOH Analyzer provide comprehensive methodology to do it. I downloaded an example of the IOH usage, iohtutorial.ipynb. It includes getting the problem, adding custom function,

For the get_problem function, there is already one get_problem function in IOH Profiler.




