# New Idea for GA

- calculate 5 shortests paths (lowest routing cost) for each demand
- a DNA in GA consists of |E| integers plus 5*|D| floats 
- |E| integers represent the chosen module on an edge, and the floats represent the demand flow on a path
- use about 1000 individuals at first, set upper limit to 5000
- 


## How GA works

### Required Definitions

- Chromosome Structure (Individual)
- Individual Initialization Function
- Mutation Operator
- Crossover Operator
- Fitness Function
- Main GA loop