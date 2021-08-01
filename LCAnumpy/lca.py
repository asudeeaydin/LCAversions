#
#
# Jesse Livezey 2014-04-19
#


import numpy as np

#Initialize settings for inference
def infer(basis, stimuli, eta, lamb, nIter, adapt, coeffs=None, softThresh=0):
    """Infers sparse coefficients for dictionary elements when representing a stimulus using LCA algorithm.

        Args:
            basis: Dictionary used to represent stimuli. Should be arranged along rows.
            coeffs: Values to start pre-threshold dictionary coefficients at for all stimuli.
            stimuli: Goals for dictionary representation. Should be arranged along rows.
            eta: Controls rate of inference. Equals to 1/tau in 2018 Olshausen paper.
            thresh: Threshold used in calculation of output variable of model neuron.
            lamb: Minimum value for thresh.
            nIter: Numer of times to run inference loop.
            softThresh: Boolean choice of threshold type.
            adapt: Amount to change thresh by per run.

        Results:
            a: Post-threshold dictionary coefficients.
            u: Pre-threshold internal *voltage.*
            thresh: Final value of thresh variable.
                                                                                                                            
        Raises:
        """
    numDict = basis.shape[0] # number of elements in dictionary
    numStim = stimuli.shape[0] # number of stimuli
    dataSize = basis.shape[1] # size of a dictionary element

    #Initialize u and a
    u = np.zeros((numStim, numDict))
    # Don't understand what this does yet
    if coeffs is not None:
        u[:] = np.atleast_2d(coeffs)
    a = np.zeros_like(u)
    ci = np.zeros((numStim, numDict))

    # Calculate G: overlap of basis functions with each other minus identity
    # Row-wise correlation matrix - identity matrix to eliminate self correlation
    G = basis.dot(basis.T) - np.eye(numDict)

    #b[i,j] is the overlap from stimuli:i and basis:j
    b = stimuli.dot(basis.T)

    thresh = np.absolute(b).mean(1)

    #Update u[i] and a[i] for nIter time steps
    for kk in range(nIter):
        #Calculate ci: amount other neurons are stimulated times overlap with rest of basis
        ci[:] = a.dot(G)
        # Update u using Rozell et al. (2008) eqn.
        u[:] = eta*(b-ci)+(1-eta)*u

        if softThresh == 1:
            # shrinkage with thresh
            a[:] = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis])
        else: # hard thresh
            a[:] = u
            # Converts 'thresh' from 1D vector to 2D where each element is a row value.
            # Compares every element of a row of 'a' with the element of the same row in 'thresh'.
            # Hard threshold
            a[np.absolute(a) < thresh[:,np.newaxis]] = 0.

        # Multiply threshold values bigger than 'lamb' with 'adapt' to change thresh per run
        thresh[thresh>lamb] = adapt*thresh[thresh>lamb]

        # Soft thresholding - asude
        # a[:] = np.sign(u) * np.maximum(0., np.absolute(u) - lamb)

    return (a,u)
