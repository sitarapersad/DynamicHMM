# nonstationary_hmm.py

import math
import numpy
import torch
from icecream import ic

from pomegranate.hmm.dense_hmm import DenseHMM
from pomegranate._utils import _cast_as_tensor, _cast_as_parameter, _check_parameter, _update_parameter

NEGINF = float("-inf")
inf = float("inf")


# TODO: Test non uniform starts and ends
# TODO: Normalize self.ends so that they are either 1 or 0; since we are assuming fixed length sequences, states can
#       only transition to `end` at the end of the sequence.

# TODO: Implement forward backward algorithm for non-stationary HMMs

def _check_inputs_base(model, X, emissions, priors):
    if X is None and emissions is None:
        raise ValueError("Must pass in one of `X` or `emissions`.")

    emissions = _check_parameter(_cast_as_tensor(emissions), "emissions",
                                 ndim=3)

    if emissions is None:
        emissions = model._emission_matrix(X, priors=priors)

    return emissions


def _check_inputs(model, X, emissions, priors):
    """
    Checks that the inputs to the forward and backward algorithms are of the correct shape and type.
    Inside this function, we call the parent class's _check_inputs method, which checks that the inputs are of the
    correct type and shape for a stationary HMM. We then check that the emissions tensor is of the correct shape for
    a non-stationary HMM.

    Note that if the model has not been initialized, the parent class's _check_inputs method will call _emission_matrix()
    which initializes the model.

    :param model:
    :param X:
    :param emissions:
    :param priors:
    :return:
    """
    # from pomegranate.hmm._base import _check_inputs as _check_inputs_base
    print(X.shape, 'check inputs')
    emissions = _check_inputs_base(model, X, emissions, priors)
    # Check that emissions is the correct shape
    assert emissions.shape[1] == model.sequence_length, f'`emissions.shape[1]` ({emissions.shape[1]}) does not match ' \
                                                        f' expected sequence length ({model.sequence_length}). '
    assert emissions.shape[2] == model.n_distributions, f'`emissions.shape[2]` ({emissions.shape[2]} does not match ' \
                                                        f' the number of states ({model.n_distributions}). '
    return emissions


class DynamicHMM(DenseHMM):
    """
    Implements a non-stationary HMM which allows for the transition matrix to vary over time/sequence position.
    The sequence length must be specified at initialization.

    Uses a dense transition matrix, which is a torch tensor of size (n_states, n_states, sequence_length).
    Inspired by the pomegranate library: www.github.com/jmschrei/pomegranate

    Parameters
    ----------
    sequence_length: int (required)
        The length of the sequence to be modeled. This is used to ensure the
        non-stationary transition matrix is of the correct length and has an
        entry for each time-step.
    distributions: tuple or list
        A set of distribution objects. These objects do not need to be
        initialized, i.e., can be "Normal()".
    edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k,sequence_length-1), optional
        A dense transition matrix of probabilities for how each node or
        distribution passed in connects to each other one. The entry at edges[t] is the
        transition matrix from time step t to t+1. This must have length sequence_length-1.
    starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
        The probability of starting at each node. If not provided, assumes
        these probabilities are uniform.
    ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
        The probability of ending at each node. If not provided, assumes
        these probabilities are uniform.
    init: str, optional
		The initialization to use for the k-means initialization approach.
		Default is 'first-k'. Must be one of:
			'first-k': Use the first k examples from the data set
			'random': Use a random set of k examples from the data set
			'submodular-facility-location': Use a facility location submodular
				objective to initialize the k-means algorithm
			'submodular-feature-based': Use a feature-based submodular objective
				to initialize the k-means algorithm.
	max_iter: int, optional
		The number of iterations to do in the EM step, which for HMMs is
		sometimes called Baum-Welch. Default is 10.
	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 0.1.
    inertia: float, [0, 1], optional
        Indicates the proportion of the update to apply to the parameters
        during training. When the inertia is 0.0, the update is applied in
        its entirety and the previous parameters are ignored. When the
        inertia is 1.0, the update is entirely ignored and the previous
        parameters are kept, equivalently to if the parameters were frozen.
    frozen: bool, optional
        Whether all the parameters associated with this distribution are frozen.
        If you want to freeze individual parameters, or individual values in those
        parameters, you must modify the `frozen` attribute of the tensor or
        parameter directly. Default is False.
    check_data: bool, optional
        Whether to check properties of the data and potentially recast it to
        torch.tensors. This does not prevent checking of parameters but can
        slightly speed up computation when you know that your inputs are valid.
        Setting this to False is also necessary for compiling. Default is True.
    """

    def __init__(self, sequence_length, distributions=None, edges=None, starts=None, ends=None,
                 init='random', max_iter=1000, tol=0.1, sample_length=None,
                 return_sample_paths=False, inertia=0.0, frozen=False, check_data=True,
                 random_state=None, verbose=False):

        super().__init__(distributions=distributions, starts=starts, ends=ends,
                         init=init, max_iter=max_iter, tol=tol, sample_length=sample_length,
                         return_sample_paths=return_sample_paths, inertia=inertia,
                         frozen=frozen, check_data=check_data, random_state=random_state,
                         verbose=verbose)

        self.name = "DynamicHMM"
        self.sequence_length = sequence_length

        n = len(distributions) if distributions is not None else 0

        if edges is not None:
            self.edges = _cast_as_parameter(torch.log(_check_parameter(_cast_as_tensor(edges),
                                                                       "edges", ndim=3,
                                                                       shape=(sequence_length - 1, n, n),
                                                                       min_value=0., max_value=1.)),
                                            dtype=self.dtype)

        self._initialized = (self.distributions is not None and
                             self.starts is not None and self.ends is not None and
                             self.edges is not None and
                             all(d._initialized for d in self.distributions))

        if self._initialized:
            self.distributions = torch.nn.ModuleList(
                self.distributions)

        self._reset_cache()

    def _reset_cache(self):
        """Reset the internally stored statistics.

        This method is meant to only be called internally. It resets the
        stored statistics used to update the model parameters as well as
        recalculates the cached values meant to speed up log probability
        calculations.
        """

        if self._initialized == False:
            return

        for node in self.distributions:
            node._reset_cache()

        self.register_buffer("_xw_sum", torch.zeros(self.sequence_length - 1, self.n_distributions,
                                                    self.n_distributions, dtype=self.dtype, requires_grad=False,
                                                    device=self.device))

        self.register_buffer("_xw_starts_sum", torch.zeros(self.n_distributions,
                                                           dtype=self.dtype, requires_grad=False, device=self.device))

        self.register_buffer("_xw_ends_sum", torch.zeros(self.n_distributions,
                                                         dtype=self.dtype, requires_grad=False, device=self.device))

    def add_edge(self, start, end, prob, time_step=None):
        """Add an edge to the model.
        This method will fill in an entry in the dense transition matrix
        at row indexed by the start distribution and the column indexed
        by the end distribution. The value that will be included is the
        log of the probability value provided. Note that this will override
        values that already exist, and that this will initialize a new
        dense transition matrix if none has been passed in so far.
        Parameters
        ----------
        start: torch.distributions.distribution
            The distribution that the edge starts at.
        end: torch.distributions.distribution
            The distribution that the edge ends at.
        prob: float, (0.0, 1.0]
            The probability of that edge.
        time_step: int, optional
            The time step to add the edge at. If not provided, will add
            the edge at every time step.
        """

        print('Adding edge')
        if self.distributions is None:
            raise ValueError("Must add distributions before edges.")

        if self._initialized:
            raise ValueError("Cannot add edges after initialization.")

        n = self.n_distributions

        if start == self.start:
            if self.starts is None:
                self.starts = torch.empty(n, dtype=self.dtype,
                                          device=self.device) - inf

            idx = self.distributions.index(end)
            self.starts[idx] = math.log(prob)

        elif end == self.end:
            if self.ends is None:
                self.ends = torch.empty(n, dtype=self.dtype,
                                        device=self.device) - inf

            idx = self.distributions.index(start)
            self.ends[idx] = math.log(prob)

        else:
            if self.edges is None:
                self.edges = torch.empty((self.sequence_length - 1, n, n), dtype=self.dtype,
                                         device=self.device) - inf

            idx1 = self.distributions.index(start)
            idx2 = self.distributions.index(end)

            if time_step is None:
                self.edges[:, idx1, idx2] = math.log(prob)
            else:
                assert time_step < self.sequence_length - 1, 'Time step must be less than sequence length - 1. Time ' \
                                                             'step t corresponds to the transition from t to t+1. '
                self.edges[time_step, idx1, idx2] = math.log(prob)

    def _initialize(self, X=None, sample_weight=None):
        """Initialize the probability distribution.
        This method is meant to only be called internally. It initializes the
        parameters of the distribution and stores its dimensionality. For more
        complex methods, this function will do more.
        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d), optional
            The data to use to initialize the model. Default is None.
        sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
            A set of weights for the examples. This can be either of shape
            (-1, len) or a vector of shape (-1,). If None, defaults to ones.
            Default is None.
        """

        print('Somehow calling initialize')
        # Important to do this first so that a non-stationary transition matrix is default
        n = self.n_distributions
        if self.edges is None:
            self.edges = _cast_as_parameter(torch.log(torch.ones(self.sequence_length - 1, n, n,
                                                                 dtype=self.dtype, device=self.device) / n))
            # Since this is a non-stationary HMM, we want to initialize the end distribution so that every state has
            # either a 0 or 1 probability of ending at the end of the sequence.
        if self.ends is None:
            # We want to initialize the end distribution so that every state has the same probability (1) of ending
            self.ends = _cast_as_parameter(torch.log(torch.ones(n, dtype=self.dtype, device=self.device)))
        super()._initialize(X, sample_weight=sample_weight)

        # Since this is a non-stationary HMM, we want to normalize the end distribution so that every state has either
        # a 0 or 1 probability of ending at the end of the sequence. Therefore, if the end probability has non-zero
        # weight (logprob not equal to -inf), we normalize it to 1.0 (logprob 0.0).
        self.ends[~torch.isinf(self.ends)] = 0.

        self.distributions = torch.nn.ModuleList(self.distributions)

    def forward(self, X=None, emissions=None, priors=None, scaling: bool = True):
        """Run the forward algorithm on some data.
        Runs the forward algorithm on a batch of sequences. This is not to be
        confused with a "forward pass" when talking about neural networks. The
        forward algorithm is a dynamic programming algorithm that begins at the
        start state and returns the probability, over all paths through the
        model, that result in the alignment of symbol i to node j.
        Note that, as an internal method, this does not take as input the
        actual sequence of observations but, rather, the emission probabilities
        calculated from the sequence given the model.

        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
            A set of examples to evaluate. Does not need to be passed in if
            emissions are.
        emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_dists)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.
        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.
        scaling: bool, optional
            Whether or not to use scaling in the forward algorithm. This is
            recommended for longer sequences to prevent precision issues when
            multiplying long sequences of probabilities. Default is True.

        Returns
        -------
        f: torch.Tensor, shape=(batch_size, self.sequence_length, self.n_distributions)
            The log probabilities calculated by the forward algorithm.
        lp: torch.Tensor, shape=(batch_size,)
            The log probability of the each observation under the model.
        """
        emissions = _check_inputs(self, X, emissions, priors)

        # Emission is of shape (batch_size, self.sequence_length, self.n_distributions)
        # emissions[i,j,k] is the log-probability of sample i at time j coming from distribution k
        batch_size = emissions.shape[0]

        # Normalize each row of the transition matrix
        T = torch.exp(self.edges - torch.logsumexp(self.edges, axis=-1, keepdims=True))

        # f is of shape (self.sequence_length, batch_size, self.n_distributions)
        # f[i,j,k] is the log-probability of sample j at time i coming from distribution k
        f = torch.clone(emissions.permute(1, 0, 2)).contiguous()

        # We will normalize the forward probabilities to prevent underflow; track the scaling factors
        # which are the log of the normalization constants for each observation at a given time step
        scaling_factors = torch.zeros((self.sequence_length, batch_size, 1), device=self.device)

        f[0] += self.starts
        scaling_factors[0] = torch.logsumexp(f[0], axis=-1, keepdims=True)
        f[0] -= scaling_factors[0]

        for i in range(1, self.sequence_length):
            # f[t] is of shape (batch_size, self.n_distributions)
            # f[t] contains the forward probability of all sequences at time t under each distribution

            t = T[i - 1]
            p_max = torch.max(f[i - 1], dim=1, keepdims=True).values
            p = torch.exp(f[i - 1] - p_max)
            f[i] += torch.log(torch.matmul(p, t)) + p_max

            # Compute and store the normalizing constant for each observation (sum over all states)
            scale_factor = torch.logsumexp(f[i], axis=-1, keepdims=True)
            # Normalize the forward probabilities
            f[i] -= scale_factor
            # Store the scaling factor for this time step
            scaling_factors[i] = scale_factor

        f += torch.cumsum(scaling_factors, dim=0)
        # Finally, we incorporate sum over transitions from the start state to time step t=0 to get the
        # overall sequence probability.
        lp = torch.logsumexp(f[-1] + self.ends, dim=1)

        # Return to original shape
        f = f.permute(1, 0, 2)

        # Check that the forward matrix has correct dimensions
        assert f.shape[1] == self.sequence_length
        assert f.shape[2] == self.n_distributions

        return f, lp

    def backward(self, X=None, emissions=None, priors=None, scaling: bool = True):
        """Run the backward algorithm on some data.
        Runs the backward algorithm on a batch of sequences. This is not to be
        confused with a "backward pass" when talking about neural networks. The
        backward algorithm is a dynamic programming algorithm that begins at end
        of the sequence and returns the probability, over all paths through the
        model, that result in the alignment of symbol i to node j, working
        backwards.
        Note that, as an internal method, this does not take as input the
        actual sequence of observations but, rather, the emission probabilities
        calculated from the sequence given the model.

        The recursive formula for the backward algorithm is:
        beta_{t}(i) = sum_{j=1}^{N} T[t]_{ij} e_{j}(x_{t+1}) beta_{t+1}(j)
        where
            N is the number of hidden states,
            T[t]_{ij} is the transition probability from state i at time t to state j at time t+1,
            e_{j}(x_{t+1}) is the probability of observing x_{t+1} given that the hidden state is j, and
            beta_{t+1}(j) is the probability of observing the sequence x_{t+2}, x_{t+2}, ..., x_{T} given that the
            hidden state at time t+1 is j
        The initial condition is beta_{T}(i) = end_distribution[i] for all i. This is generally set to 1 for all i.

        Note: For a sequence of length L, we have L-1 'real' transitions and an implicit transition from the silent
        start state at time t=-1 to first real hidden state at time t=0. The transition matrix for this implicit
        transition is parametrized by the start distribution.

        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
            A set of examples to evaluate. Does not need to be passed in if
            emissions are.
        emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.
        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.
        scaling: bool, optional
            Whether or not to use scaling in the forward algorithm. This is
            recommended for longer sequences to prevent precision issues when
            multiplying long sequences of probabilities. Default is True.
        Returns
        -------
        b: torch.Tensor, shape=(-1, length, self.n_distributions)
            The log probabilities calculated by the backward algorithm.
        lp: torch.Tensor, shape=(batch_size,)
            The log probability of the each sequence under the model.
        """

        emissions = _check_inputs(self, X, emissions, priors)
        batch_size = emissions.shape[0]

        # We permute emissions so that the sequence length is the first dimension for iteration
        emissions = emissions.permute(1, 0, 2)

        # Normalize each row of the transition matrix
        T = torch.exp(self.edges - torch.logsumexp(self.edges, axis=-1, keepdims=True))

        # We will normalize the backward probabilities to prevent underflow; track the scaling factors
        # which are the log of the normalization constants for each observation at a given time step
        scaling_factors = torch.zeros((self.sequence_length, batch_size, 1), device=self.device)

        # We will store the backward probabilities in a tensor of shape (sequence_length, batch_size, n_distributions)
        b = torch.ones(self.sequence_length, batch_size, self.n_distributions, dtype=self.dtype,
                       device=self.device) + float("-inf")

        # self.ends[i] specifies the probability of transitioning from state i to the end state, which is observed with
        # probability 1 after the 'real' sequence.
        b[-1] = self.ends
        scaling_factors[-1] = torch.logsumexp(b[-1], axis=-1, keepdims=True)
        b[-1] -= scaling_factors[-1]

        for i in range(self.sequence_length - 2, -1, -1):
            # For each time step, we first sum the log-probability of the emissions of observation n at
            # time step i+1 (e[i+1] with shape n x self.n_distributions) with b[i+1] (the log of the
            # backwards probabilities at time step i+1 with shape n x self.n_distributions).
            be = b[i + 1] + emissions[i + 1]

            # We multiply this with the transition matrix T[i] (with shape self.n_distributions x self.n_distributions)
            # and sum over the rows to get the log of the backwards probabilities at time step i.
            b[i] = torch.log(torch.matmul(torch.exp(be), T[i].T))

            # Compute and store the normalizing constant for each observation (sum over all states)
            scale_factor = torch.logsumexp(b[i], axis=-1, keepdims=True)
            # Normalize the forward probabilities
            b[i] -= scale_factor
            # Store the scaling factor for this time step
            scaling_factors[i] = scale_factor

        # If we are using scaling, we need to incorporate the scaling factors into the overall sequence probability.
        # We need to compute the cumulative sum of the scaling factors for each observation in reversed time order
        # (i.e. from the end of the sequence to the beginning).
        b += torch.flip(torch.cumsum(torch.flip(scaling_factors, [0]), dim=0), [0])

        # Finally, we incorporate sum over transitions from the start state to time step t=0 to get the
        # overall sequence probability.
        lp = torch.log(torch.matmul(torch.exp(b[0] + emissions[0]), torch.exp(self.starts)))

        b = b.permute(1, 0, 2)

        # Check that the forward matrix has correct dimensions
        assert b.shape[1] == self.sequence_length
        assert b.shape[2] == self.n_distributions

        return b, lp

    def forward_backward(self, X=None, emissions=None, priors=None):
        """Run the forward-backward algorithm on some data.
        Runs the forward-backward algorithm on a batch of sequences. This
        algorithm combines the best of the forward and the backward algorithm.
        It combines the probability of starting at the beginning of the sequence
        and working your way to each observation with the probability of
        starting at the end of the sequence and working your way backward to it.
        A number of statistics can be calculated using this information. These
        statistics are powerful inference tools but are also used during the
        Baum-Welch training process.

        See https://web.stanford.edu/~jurafsky/slp3/A.pdf for more information.

        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
            A set of examples to evaluate. Does not need to be passed in if
            emissions are.
        emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.
        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.
        Returns
        -------
        transitions: torch.Tensor, shape=(-1, n, n)
            The expected number of transitions across each edge that occur
            for each example. The returned transitions follow the structure
            of the transition matrix and so will be dense or sparse as
            appropriate.
        responsibility: torch.Tensor, shape=(-1, -1, n)
            The posterior probabilities of each observation belonging to each
            state given that one starts at the beginning of the sequence,
            aligns observations across all paths to get to the current
            observation, and then proceeds to align all remaining observations
            until the end of the sequence.
        starts: torch.Tensor, shape=(-1, n)
            The probabilities of starting at each node given the
            forward-backward algorithm.
        ends: torch.Tensor, shape=(-1, n)
            The probabilities of ending at each node given the forward-backward
            algorithm.
        logp: torch.Tensor, shape=(-1,)
            The log probabilities of each sequence given the model.
        """

        emissions = _check_inputs(self, X, emissions, priors)
        n, l, _ = emissions.shape

        f, lp_f = self.forward(emissions=emissions)
        b, lp_b = self.backward(
            emissions=emissions)  # b[obs,time,state] = backward probability of state at time for observation number obs

        logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)

        # For epsilons, we need to compute for every observation, a value for each (start state, end state, time step)
        f_ = f[:, :-1].unsqueeze(-1)
        b_ = (b[:, 1:] + emissions[:, 1:]).unsqueeze(-2)
        t = f_ + b_ + self.edges

        # t[0][t][i][j] contains the 'almost' log of the expected counts of transitions from state i to state j at time t.
        # We still need to normalize by the probability of the observation at time t, which is the sum of the probabilities of being in each state at time t.

        # obs_logprob[0][t] contains the log of the probability of the observation at time t.
        obs_logprob = torch.logsumexp((f[:, :-1] + b[:, :-1]), axis=2)
        t = torch.exp(t - obs_logprob.unsqueeze(2).unsqueeze(2))

        # We also update the transitions from the silent start state to time 0
        starts = self.starts + emissions[:, 0] + b[:, 0]
        starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T

        # We also update the transitions from end of the sequence to the silent end state
        # For a non-stationary HMM, all states transitions to the end state at the end of the sequence
        # with probability 0 or 1.
        ends = self.ends + f[:, -1]
        ends[~torch.isinf(ends)] = 0.
        ends = torch.exp(ends)

        # Finally, we compute gamma, the probability of being in state i at time t given the entire sequence
        gamma = f + b
        gamma = gamma - torch.logsumexp(gamma, dim=2).reshape(n, -1, 1)

        return t, gamma, starts, ends, logp

    def viterbi(self, X=None, emissions=None, priors=None):
        """Run the Viterbi decoding algorithm on some data.

        Run the Viterbi algorithm on the sequence given the model. This finds the ML path of hidden states given the
        sequence. Returns a tuple of the log probability of the ML path, or (-inf, None) if the sequence is
        impossible under the model. If a path is returned, it is a list of tuples of the form (sequence index,
        state object).

        This is fundamentally the same as the forward algorithm using max instead of sum. Note that this is not
        implemented for Silent states.

        Parameters
        ----------
        X: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)
            A set of examples to evaluate. Does not need to be passed in if
            emissions are.
        emissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.
        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.
        Returns
        -------
        logprob: torch.Tensor, shape=(batch_size,)
            The log probability of the most likely path through the model for each example.
        path: torch.Tensor, shape=(batch_size, self.sequence_length)
            The most likely path through the model for each example.
        viterbi: torch.Tensor, shape=(batch_size, self.sequence_length, self.n_distributions)
            The viterbi matrix for each example.
        """
        emissions = _check_inputs(self, X, emissions, priors)
        # Emission is of shape (batch_size, self.sequence_length, self.n_distributions)
        # emissions[i,j,k] is the log-probability of sample i at time j coming from distribution k
        batch_size = emissions.shape[0]

        # We permute emissions so that the sequence length is the first dimension for iteration
        emissions = emissions.permute(1, 0, 2)

        # Normalize each row of the transition matrix. NB. This is different from the forward/backward algorithms
        # since we stay in the log space.
        T = self.edges - torch.logsumexp(self.edges, axis=-1, keepdims=True)

        v = torch.zeros(self.sequence_length, batch_size, self.n_distributions, dtype=self.dtype)
        # v[t,n,k] is the log-probability of the best path through sample n at time t ending in state k
        traceback = torch.zeros(self.sequence_length, batch_size, self.n_distributions, dtype=torch.long)

        v[0] += self.starts + emissions[0]
        for i in range(1, self.sequence_length):
            t = T[i - 1]
            e = emissions[i]

            # The entry at position (n, k', k) is the score of the best path in sequence n which ends in state k and
            # is immediately preceded by state k'
            v_i_options = ((v[i - 1][:, :, None] + t) + e[:, None, :])

            # Now we want to find the best path ending in each state and the score of that path
            # We store the best predecessor k' for each state k
            v[i], traceback[i] = torch.max(v_i_options, axis=1)

        # Return to original shape
        v = v.permute(1, 0, 2)

        # Check that the forward matrix has correct dimensions
        assert v.shape[1] == self.sequence_length
        assert v.shape[2] == self.n_distributions

        lp, final_states = v[:, -1, :].max(axis=1)
        final_state = final_states.long()
        # Track most recent best state for each observation
        mrs = final_state
        path = [final_state]
        for i in range(self.sequence_length - 1, 0, -1):
            # This is a matrix of dimension (batch_size, self.n_distributions)
            # The entry at TB[i][n,k] shows the best previous state for sample n at time i ending in state k
            mrs = traceback[i][torch.arange(len(mrs)), mrs]
            path.append(mrs)
        path = torch.stack(path[::-1], dim=1).int()

        return lp, path, v

    def summarize(self, X, sample_weight=None, emissions=None, priors=None):
        """Extract the sufficient statistics from a batch of data.

        This method calculates the sufficient statistics from optionally
        weighted data and adds them to the stored cache. The examples must be
        given in a 2D format. Sample weights can either be provided as one
        value per example or as a 2D matrix of weights for each feature in
        each example.


        Parameters
        ----------
        X: torch.Tensor, shape=(-1, -1, self.d)
            A set of examples to summarize.

        y: torch.Tensor, shape=(-1, -1), optional

            A set of labels with the same number of examples and length as the
            observations that indicate which node in the model that each
            observation should be assigned to. Passing this in means that the
            model uses labeled training instead of Baum-Welch. Default is None.

        sample_weight: torch.Tensor, optional
            A set of weights for the examples. This can be either of shape
            (-1, self.d) or a vector of shape (-1,). Default is ones.

        emissions: torch.Tensor, shape=(-1, -1, self.n_distributions)
            Precalculated emission log probabilities. These are the
            probabilities of each observation under each probability
            distribution. When running some algorithms it is more efficient
            to precalculate these and pass them into each call.

        priors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)
            Prior probabilities of assigning each symbol to each node. If not
            provided, do not include in the calculations (conceptually
            equivalent to a uniform probability, but without scaling the
            probabilities). This can be used to assign labels to observatons
            by setting one of the probabilities for an observation to 1.0.
            Note that this can be used to assign hard labels, but does not
            have the same semantics for soft labels, in that it only
            influences the initial estimate of an observation being generated
            by a component, not gives a target. Default is None.
        """

        # The parent class DenseHMM overwrites BaseHMM in a way we do not want; we need to call the summarize method of
        # the "grandparent" class
        X, emissions, sample_weight = super(DenseHMM, self).summarize(X,
                                                                      sample_weight=sample_weight,
                                                                      emissions=emissions,
                                                                      priors=priors)

        t, r, starts, ends, logps = self.forward_backward(emissions=emissions)

        # We need to store the sum of the weights of the examples that end at each state
        self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
        self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
        self._xw_sum += torch.sum(t * sample_weight.unsqueeze(-1).unsqueeze(-1), dim=0)

        X = X.reshape(-1, X.shape[-1])
        r = torch.exp(r) * sample_weight.unsqueeze(-1)
        for i, node in enumerate(self.distributions):
            w = r[:, :, i].reshape(-1, 1)
            node.summarize(X, sample_weight=w)

        return logps

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

        This method uses calculated statistics from calls to the `summarize`
        method to update the distribution parameters. Hyperparameters for the
        update are passed in at initialization time.

        Note: Internally, a call to `fit` is just a successive call to the
        `summarize` method followed by the `from_summaries` method.
        """

        # Update emission distributions
        for node in self.distributions:
            node.from_summaries()

        if self.frozen:
            return

        # For a non-stationary HMM, all states transitions to the end state at the end of the sequence
        # with probability 0 or 1.
        ends = torch.log(self._xw_ends_sum > 0)
        starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
        # Normalize the transition matrix so that each row sums to 1
        edges = torch.log(self._xw_sum / self._xw_sum.sum(axis=-1).unsqueeze(-1))

        _update_parameter(self.ends, ends, inertia=self.inertia)
        _update_parameter(self.starts, starts, inertia=self.inertia)
        _update_parameter(self.edges, edges, inertia=self.inertia)
        self._reset_cache()

    def sample(self, n_samples):
        """
        Sample a sequence from the model. We start at the start state and sample from the distribution at that state.
        We then sample from the transition matrix to get the next state and sample from the distribution at that state.
        We repeat this process until we reach the end of the sequence.

        Parameters
        ----------
        n_samples: int
            The number of samples to generate.
        Returns
        -------
        X: torch.Tensor, shape=(n_samples, self.sequence_length, self.d)
            The generated samples.
        viterbi: torch.Tensor, shape=(n_samples, self.sequence_length)
            The state path through the model for each example.
        """
        X = torch.zeros((n_samples, self.sequence_length, self.d), dtype=self.dtype, device=self.device)
        viterbi = torch.zeros((n_samples, self.sequence_length), dtype=self.dtype,
                              device=self.device)

        for _ in range(n_samples):
            state = torch.multinomial(torch.exp(self.starts), 1).item()
            for i in range(self.sequence_length):
                X[_, i] = self.distributions[state].sample(1)
                viterbi[_, i] = state
                if i < self.sequence_length - 1:
                    state = torch.multinomial(torch.exp(self.edges[i, state]), 1).item()
        return X, viterbi
