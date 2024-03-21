# DynamicHMM
Implements non-stationary HMM model (based on pomegranate's DenseHMM class)

<hr>
Implements a non-stationary HMM which allows for the transition matrix to vary over time/sequence position.
The sequence length must be specified at initialization.

Uses a dense transition matrix, which is a torch tensor of size (n_states, n_states, sequence_length).
Inspired by the pomegranate library: www.github.com/jmschrei/pomegranate
    
