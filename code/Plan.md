## Next things to work on:
- Implement policy gradient descent
- Setup logic for artificially inflated input / output dimensions, which keeps the underlying dynamics simpmle

I think the following process should be something along the lines of:
- I need to somehow evaluate performance, robustness and parameter efficiency for my controllers
- I need to implement the learning of controllers for low-rank feedback matrices (based on the papers like structured policy optimization or data enabled optimization)
- I need to artificially inflate the inputs of my current double integrator example to extend the setting to reinforcement learning (using a camera to capture the carts position, different force application points), to extend my previous learning structures with this
- I apply this new setting to all of my previous learning algorithms, and apply the evaluation metrics to this


Things to discuss:
Marked with SHOW