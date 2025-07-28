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

## Modifications

#### Elitism & Special initialization
- `deap.algorithms.eaMuPlusLambda` does elitism, meaning selects mu best individuals from offspring, and lambda best individuals from initial pop
- initializing population with individuals that have some individuals with no penalties might be good

#### Better Crossover
- What if instead of doing crossover totally randomly, we create a random iteration over the demands (not from first to last always), and then evaluate one for one which flow distribution of the which parent would have lower cost (including penalties once overflow)