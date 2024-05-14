import numpy as np


def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2 * L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode="constant")

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n : n + M, n : n + M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N - L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov


def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1 / 2) / (L * var)
    axis = np.arange(-L, L + 1)
    gaussian1D = np.exp(-(taper**2) * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def compute_kernel_checkerboard_box(L):
    """Compute box-like checkerboard kernel [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size 2*L+1

    Returns:
        kernel (np.ndarray): Kernel matrix of size (2*L+1) x (2*L+1)
    """
    axis = np.arange(-L, L + 1)
    kernel = np.outer(np.sign(axis), np.sign(axis))
    return kernel


def calculate_correlation(sequence1, sequence2):
    # make sequences same length
    if len(sequence1) > len(sequence2):
        sequence1 = sequence1[: len(sequence2)]
    elif len(sequence2) > len(sequence1):
        sequence2 = sequence2[: len(sequence1)]
    return np.nan_to_num(np.corrcoef(sequence1, sequence2)[0][1])


def kronecker_delta(x, y):
    if x == y:
        return 1
    else:
        return 0


# Create similarity matrix from MIDI note numbers
def calculate_ssm(sequence, similarity_function):
    length = len(sequence)
    ssm = np.zeros((length, length))

    ## STUDENT SECTION - REMOVE ALL OR SOME OF CODE ##
    i = 0
    for element_i in sequence:
        j = 0
        for element_j in sequence:
            ssm[i][j] = similarity_function(element_i, element_j)
            j += 1
        i += 1
    ## END STUDENT SECTION  ##

    return ssm


def get_novelty_topk(novelty, k=10):
    return np.sort(np.argsort(novelty)[-k:])


def get_novelty_threshold(novelty, threshold):
    mask = novelty > threshold
    arange = np.arange(novelty.shape[0])
    return arange[mask]
