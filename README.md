# Neural Architechture Search

# Hardware Aware Neural Architecture Search (NAS) for NNabla
This toolbox provides methods for neural hardware aware neural architecture search 
for NNabla, i.e., it implements tools for

1. A top level graph to define candidate architectures for convolutional neural networks (CNNs).
2. Profilers to measure the hardware demands of neural architectures (latency, #parameters, ...).
3. Learners to update the architecture and model parameters (DARTS, ProxylessNAS).
4. Regularizers which can be used to enforce hardware constraints.
5. Tools for visualization and debugging.

This package has been tested, using the environment
described [here](/environment.txt).

## Neural Architecture Search
### Basics
Neural Architecture Search (NAS) is a subfield of Deep Learning which
deals with approaches to automatically learn the optimal network architecture
to solve a given machine learning task. A DNN architecture defines 
how single DNN layers are connected to each other to form a network. 
Mathematically, the architecture defines how the transfer 
functions $`\{f_l(\underline{x}, \underline{\theta}_l)\}_{l=1}^{L}`$ of $`L`$ 
layers are concatenated  to form a global transfer function $`f(\underline{x}, \underline{\theta})`$ 
of the DNN, meaning that it defines the family of transfer functions a 
DNN can parametrize by changing the parameters $`\underline{\theta}`$.
A list of selected papers, which describe the following paragraphs in detail is
given [here](/theory/papers/bibliography.bib).

The design of modern Neural Network architectures is driven by multiple different objectives: 
1. First,  the network should have a reasonably high capacity, meaning that
.the the family of transfer functions contains arbitrary 
complex functions which can capture lots of information from training data. 
Therefore, we long for networks layers which can parametrize very complex 
input-to-output mappings and have a large number of parameters. 
2. Second, we want to assure that the network can be trained well, 
.meaning for example that gradient backpropagation 
should not suffer from vanishing gradients. 
3. Third, we want to choose an architecture with a strong a inductive bias
for the given target task. A DNN with such an architecture defines 
a family of transfer functions which needs only little training 
data and generalize well to unseen data. 
If we for example know that the network output should be invariant 
to a certain transformation of the input data, we can use this knowledge 
and use layers which naturally lead to such an invariance. 
In case of image classification, we for example know that the DNN used for this task should 
be invariant to shifts of the input image. This can be achieved by concatenating 
convolutional and pooling layers. However, in general it is very difficult to 
come up with such domain knowledge. Moreover, there is no theory and only little 
empirical knowledge which combination of layers lead to DNNs with a good inductive 
bias for certain tasks. Hence, designing architectures to have good inductive 
bias boils down to a good intuition and lots of luck.
4. Last, DNN inference should also be computationally efficient, meaning that 
inference only needs a small number of multiplication-accumulation (MAC) 
operations and that the DNN has a small memory footprint. To achieve this, 
we long for DNN architectures with layers which have only a small number of
parameters and with a very simple input-to-output mapping.

The design of a good DNN architecture means to find a good balance 
between those (often competing) requirements, by selecting and arranging
layers in a meaningful way. Compared to the early days of Deep Learning, today, 
DNNs consist of a broad variety of different different network layers like:

* fully connected (affine)
* convolutional (with and without bias)
* dilated convolutional
* group convolutional
* separable convolutional (depth wise, channel wise, spatial)
* pooling
* shortcut
* batch normalization

Therefore, neural architecture design is a very large combinatorial problem which 
is especially hard to solve, because we have only a poor (or almost no) 
understanding how a specific choice or arrangement of layers effects our 
requirements. The aim of neural architecture search is to automate architecture 
design and to directly learn the optimal architecture from the data. 
This has multiple benefits. We need no expert with lots of experience. 
We do not need to understand which effect a combination of certain layers yields to our requirements. 
NAS has the potential to come up with architectures which generalize much better 
to unseen data than humans, because it can try out many more architectures in the same time.
We can optimize the architectures to be resource efficient.

To this end, NAS approaches use methods from the areas of:
Analysis, reinforcement learning, genetic algorithms, Bayesian optimization, 
Deep Neural Network Scaling, machine learning on graphs.

### The general NAS algorithm
All existing NAS algorithms are based on the same idea, show in
the figure below. Rather than performing a general search over all possible network
architectures, which would be intractable, the NAS algorithms work on a restricted
and predefined candidate space 
$`\mathcal{F}=\{f({\bf x}; {\bf \theta}_m, {\bf \theta}_a): {\bf \theta}_m \in \mathbb{R}^{d}, {\bf \theta}_a \in \mathbb{R}^{k}\}`$, 
which is the set of DNNs which is mapped out by all possible model parameter vectors 
$`{\bf \theta}_m \in \mathbb{R}^d`$ (i.e. the weights of a DNN) and all possible
architecture parameter vectors $`{\bf \theta}_a \in \mathbb{R}^k`$ (i.e. vectors,
which define what kind of layers are used in the DNN and how they are connected).
Dependent on the NAS algorithm, the candidate space $`\mathcal{F}`$ can be very 
large and may consist of all possible DNNs which can be 
constructed from a given set of layers (e.g. in case of the 
[NAS Bench 101 paper](/theory/papers/nas_bench_101.pdf)), or the candidiate space 
can be very small and restrictive 
(e.g. in case of the [DARTS paper](/theory/papers/darts.pdf)).

For a given training set $`\mathcal{T} = \{ ({\bf x}_t(n), {\bf y}_t(n) \}_{n=1}^{N_t} \}`$ 
and a given test set $`\mathcal{V} = \{ ({\bf x}_v(n), {\bf y}_v(n) \}_{n=1}^{N_v}\}`$ with 
$`N_t`$ and $`N_v`$ samples, respectively, the NAS algorithm tries to solve the 
nested optimization problem
$`\min J(f({\bf x};  {\bf \theta}^{\ast}_m, {\bf \theta}_a), \mathcal{V})`$ s.t. 
$`{\bf \theta}^{\ast}_m = \arg \min_{{\bf \theta}_m} J(f({\bf x}; {\bf \theta}_m, {\bf \theta}_a), \mathcal{T})`$,
i.e. it tries to optimize the architecture parameters $`{\bf \theta}_a`$ to
minimize the validation loss $`J(f({\bf x};{\bf \theta}^{\ast}_m, {\bf \theta}_a), \mathcal{V})`$,
under the constraint that the model parameters $`{\bf \theta}_m`$ are chosen to optimize the 
training error $`J(f({\bf x}; {\bf \theta}_m, {\bf \theta}_a), \mathcal{T})`$.

In general, solving this nested optimization problem exactly is prohibitively complex.
Therefore, all NAS algorithms simplify this optimization problem, such that
it can be solved in reasonable time. The main difference of the proposed NAS algorithms
is how they design the candidate space $`\mathcal{F}`$ and how they simplify the 
optimization problem given above.

### Hardware aware NAS
Hardware aware NAS algorithms try to solve the optimization problem given above 
under additional hardware constraints. For example, we could be interested to 
find the optimal architecture in $`\mathcal{F}`$, for which the number of model
parameters $`d`$ does not exceed a given value (a memory constraint) 
or which can be evaluated within a given time on a specific target platform 
(a latency constraints). Mathematically, this means to add more constraints to 
the optimization problem given above, i.e.,
$`\min J(f({\bf x};  {\bf \theta}^{\ast}_m, {\bf \theta}_a), \mathcal{V})`$ s.t. 
$`{\bf \theta}^{\ast}_m = \arg \min_{{\bf \theta}_m} J(f({\bf x}; {\bf \theta}_m, {\bf \theta}_a), \mathcal{T})`$,
$`C_i(f({\bf x};  {\bf \theta}_m, {\bf \theta}_a)) \leq 0, \:\: \forall i=1,...,I`$.
Here, $`C_i(f({\bf x};  {\bf \theta}_m, {\bf \theta}_a)) \leq 0`$ are $`I`$ inequality
constraints which restrict the hardware demands to evaluate the transfer 
function $`f(\cdot)`$ of the DNN. As described later in detail, 
$`C_i(f({\bf x};  {\bf \theta}_m, {\bf \theta}_a))`$ can change for different 
architecture parameters $`{\bf \theta}_a`$. However, they are independent of and
hence show no change with respect to the model parameters $`{\bf \theta}_m`$.

# The structure of the NNabla NAS framework

# Install `nnabla_nas`
```python
pip install -r requirements.txt
```

# Experiments

See `jobs.sh`
