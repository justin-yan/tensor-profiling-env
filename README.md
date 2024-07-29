# tenprof

In order to fully characterize a layer of a DNN, you'd like to graph the curves of the following values as you vary input sizes and hyperparameter selections:

- FLOPS throughput
- Memory Bandwidth consumed
- GPU Memory utilization
- Wall clock time

This should help you understand whether your bottlenecks are hardware related (e.g. if you scale your inputs and start hitting OOMs), or algorithm related (if you do a large number of small sequential operations, you'll see poor hardware utilization on all fronts).

To get these raw resource readouts, use the basic profilers:

- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://triton-lang.org/main/python-api/triton.testing.html
- https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59/7
- fvcore flop counter

It's possible that you may identify a problematic layer that you need to drop to an even lower level to understand (e.g. is it slow because your register caching strategy isn't working as expected?), at which point you'll need to drop down to using the NVidia profilers.

## Layer Profiler

Basic inner implementation:

- Given a concrete instance of a layer that is jaxtyped,
- Auto-generate inputs for the layer.

## Backlog

- For forward and backward passes (broken out):
  - Count FLOPS
  - Calculate wall clock time
  - Profile memory usage and get overall counts

- Create a small grid for each input size dimension.
  - Or follow something like property-based testing and scale up sizes until bottlenecks are reached?
- Create a small grid for different hyperparameters in the layer constructor(?)
  - Allow for partially applied layers?
- Allow for more flexible layer dtype defaults, and selecting input tensor dtypes to match.
