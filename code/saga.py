### Importing dependencies

# Gaussian sampler
from sampler import samplerz
# Gaussian sampler with repetitions
from sampler_rep import samplerz_rep
# Imports Falcon signature scheme
from falcon import falcon

# Estimators for moments
from scipy.stats import skew, kurtosis, moment
# Statistical (normality) tests
from scipy.stats import chisquare
# Distributions
from scipy.stats import chi2, norm
# Numpy stuff
from numpy import cov, set_printoptions, diag, array, mean
from numpy.linalg import matrix_rank, inv, eig, eigh
import matplotlib.pyplot as plt

# Math functions
from math import ceil, sqrt, exp, log
# Data management
from copy import deepcopy
import re
import pandas
# Uniformity
from random import uniform

# For HZ multivariate test, used in scipy.spatial.distance.mahalanobis
from numpy import floor
from numpy import tile

# qqplot
import scipy.stats as stats
from numpy import transpose, sort

# doornik hansen
from numpy import corrcoef, power
from numpy import log as nplog
from numpy import sqrt as npsqrt

# mvn plot test
from numpy import histogram

# rejection testing
from collections import Counter, defaultdict
from numpy import arange

# import csv files
import csv

# For debugging purposes
import sys
import time
if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.


# Tailcut rate
tau = 14
# Minimal size of a bucket for the chi-squared test (must be >= 5)
chi2_bucket = 10
# Minimal p-value
pmin = 0.001
# Print options
set_printoptions(precision=4)


def gaussian(x, mu, sigma):
    """
    Gaussian function of center mu and "standard deviation" sigma.
    """
    return exp(- ((x - mu) ** 2) / (2 * (sigma ** 2)))


def make_gaussian_pdt(mu, sigma):
    """
    Make the probability distribution table (PDT) of a discrete Gaussian.
    The output is a dictionary.
    """
    # The distribution is restricted to [-zmax, zmax).
    zmax = ceil(tau * sigma)
    pdt = dict()
    for z in range(int(floor(mu)) - zmax, int(ceil(mu)) + zmax):
        pdt[z] = gaussian(z, mu, sigma)
    gauss_sum = sum(pdt.values())
    for z in pdt:
        pdt[z] /= gauss_sum
    return pdt


class UnivariateSamples:
    """
    Class for computing statistics on univariate Gaussian samples.
    """

    def __init__(self, mu, sigma, list_samples):
        """
        Input:
        - the expected center mu of a discrete Gaussian over Z
        - the expected standard deviation sigma of a discrete Gaussian over Z
        - a list of samples defining an empiric distribution

        Output:
        - the means of the expected and empiric distributions
        - the standard deviations of the expected and empiric distributions
        - the skewness of the expected and empiric distributions
        - the kurtosis of the expected and empiric distributions
        - a chi-square test between the two distributions
        """
        zmax = ceil(tau * sigma)
        # Expected center standard variation.
        self.exp_mu = mu
        self.exp_sigma = sigma
        # Number of samples
        self.nsamples = len(list_samples)
        self.histogram = dict()
        self.outlier = 0
        # Initialize histogram
        for z in range(int(floor(mu)) - zmax, int(ceil(mu)) + zmax):
            self.histogram[z] = 0
        for z in list_samples:
            # Detect and count outliers (samples not in [-zmax, zmax))
            if z not in self.histogram:
                self.outlier += 1
            # Fill histogram according to the samples
            else:
                self.histogram[z] += 1
        # Empiric mean, variance, skewness, kurtosis and standard deviation
        self.mean = sum(list_samples) / self.nsamples
        self.variance = moment(list_samples, 2)
        self.skewness = skew(list_samples)
        self.kurtosis = kurtosis(list_samples)
        self.stdev = sqrt(self.variance)
        # Chi-square statistic and p-value
        self.chi2_stat, self.chi2_pvalue = self.chisquare()
        # Final assessment: the dataset is valid if:
        # - the chi-square p-value is higher than pmin
        # - there is no outlier
        self.is_valid = True
        self.is_valid &= (self.chi2_pvalue > pmin)
        self.is_valid &= (self.outlier == 0)


    def __repr__(self):
        """
        Print the sample statistics in a readable form.
        """
        rep = "\n"
        rep += "Testing a Gaussian sampler with center = {c} and sigma = {s}\n".format(c=self.exp_mu, s=self.exp_sigma)
        rep += "Number of samples: {nsamples}\n\n".format(nsamples=self.nsamples)
        rep += "Moments  |   Expected     Empiric\n"
        rep += "---------+----------------------\n"
        rep += "Mean:    |   {exp:.5f}      {emp:.5f}\n".format(exp=self.exp_mu, emp=self.mean)
        rep += "St. dev. |   {exp:.5f}      {emp:.5f}\n".format(exp=self.exp_sigma, emp=self.stdev)
        rep += "Skewness |   {exp:.5f}      {emp:.5f}\n".format(exp=0, emp=self.skewness)
        rep += "Kurtosis |   {exp:.5f}      {emp:.5f}\n".format(exp=0, emp=self.kurtosis)
        rep += "\n"
        rep += "Chi-2 statistic:   {stat}\n".format(stat=self.chi2_stat)
        rep += "Chi-2 p-value:     {pval}   (should be > {p})\n".format(pval=self.chi2_pvalue, p=pmin)
        rep += "\n"
        rep += "How many outliers? {o}".format(o=self.outlier)
        rep += "\n\n"
        rep += "Is the sample valid? {i}".format(i=self.is_valid)
        return rep

    def chisquare(self):
        """
        Run a chi-square test to compare the expected and empiric distributions
        """
        # We construct two histograms:
        # - the expected one (exp_histogram)
        # - the empirical one (histogram)
        histogram = deepcopy(self.histogram)
        # The chi-square test require buckets to have enough elements,
        # so we aggregate samples in the left and right tails in two buckets
        exp_histogram = make_gaussian_pdt(self.exp_mu, self.exp_sigma)
        obs = list(histogram.values())
        exp = list(exp_histogram.values())
        z = 0
        while(1):
            if (z >= len(exp) - 1):
                break
            while (z < len(exp) - 1) and (exp[z] < chi2_bucket / self.nsamples):
                obs[z + 1] += obs[z]
                exp[z + 1] += exp[z]
                obs.pop(z)
                exp.pop(z)
            z += 1
        obs[-2] += obs[-1]
        exp[-2] += exp[-1]
        obs.pop(-1)
        exp.pop(-1)
        exp = [round(prob * self.nsamples) for prob in exp]
        diff = self.nsamples - sum(exp_histogram.values())
        exp_histogram[int(round(self.exp_mu))] += diff
        res = chisquare(obs, f_exp=exp)
        return res


class MultivariateSamples:
    """
    Class for computing statistics on multivariate Gaussian samples
    """

    def __init__(self, sigma, list_samples):
        """
        Input:
        - sigma: an expected standard deviation
        - list_samples: a list of (expected) multivariate samples

        Output:
        - univariates[]: a list of UnivariateSamples objects (one / coordinate)
        - covariance: an empiric covariance matrix
        - DH, AS, PO, PA: statistics and p-values for the Doornik-Hansen test
        - dc_pvalue: a p-value for our custom covariance-based test
        """
        # Parse the signatures and store them
        self.nsamples = len(list_samples)
        self.dim = len(list_samples[0])
        self.data = pandas.DataFrame(list_samples)
        # Expected center and standard deviation
        self.exp_mu = 0
        self.exp_si = sigma
        # Testing sphericity
        # For each coordinate, perform an univariate analysis
        self.univariates = [None] * self.dim
        for i in range(self.dim):
            self.univariates[i] = UnivariateSamples(0, sigma, self.data[i])
        self.nb_gaussian_coord = sum((self.univariates[i].chi2_pvalue > pmin) for i in range(self.dim))
        # Estimate the (normalized) covariance matrix
        self.covariance = cov(self.data.transpose()) / (self.exp_si ** 2)
        self.DH, self.AS, self.PO, self.PA = doornik_hansen(self.data)
        self.dc_pvalue = diagcov(self.covariance, self.nsamples)

    def __repr__(self):
        """
        Print the sample statistics in a readable form.
        """
        rep = "\n"
        rep += "Testing a centered multivariate Gaussian of dimension = {dim} and sigma = {s:.3f}\n".format(dim=self.dim, s=self.exp_si)
        rep += "Number of samples: {nsamples}\n".format(nsamples=self.nsamples)
        rep += "\n"
        rep += "The test checks that the data corresponds to a multivariate Gaussian, by doing the following:\n"
        rep += "1 - Print the covariance matrix (visual check). One can also plot\n"
        rep += "    the covariance matrix by using self.show_covariance()).\n"
        rep += "2 - Perform the Doornik-Hansen test of multivariate normality.\n"
        rep += "    The p-value obtained should be > {p}\n".format(p=pmin)
        rep += "3 - Perform a custom test called covariance diagonals test.\n"
        rep += "4 - Run a test of univariate normality on each coordinate\n"
        rep += "\n"
        rep += "1 - Covariance matrix ({dim} x {dim}):\n{cov}\n".format(dim=self.dim, cov=self.covariance)
        rep += "\n"
        if (self.nsamples < 4 * self.dim):
            rep += "Warning: it is advised to have at least 8 times more samples than the dimension n.\n"
        rep += "2 - P-value of Doornik-Hansen test:                {p:.4f}\n".format(p=self.PO)
        rep += "\n"
        rep += "3 - P-value of covariance diagonals test:          {p:.4f}\n".format(p=self.dc_pvalue)
        rep += "\n"
        rep += "4 - Gaussian coordinates (w/ st. dev. = sigma)?    {k} out of {dim}\n".format(k=self.nb_gaussian_coord, dim=self.dim)
        return rep

    def show_covariance(self):
        """
        Visual representation of the covariance matrix
        """
        plt.imshow(self.covariance, interpolation='nearest')
        plt.show()

    def mardia(self):
        """
        Mardia's test of multivariate normality.

        The test compute estimators:
        - A for the "generalized skewness"
        - B for the "generalized kurtosis"

        If the data is multivariate normal, them:
        - A should follow a chi-2 distribution
        - B should follow a normal distribution

        Warning: For high dimensions, the function converges very slowly,
        requires many samples, is very slow and uses lots of memory.
        """
        if (self.nsamples < 500):
            print("At least 500 samples are recommended for Mardia's test")
        nsamp = self.nsamples
        dim = self.dim
        means = [list(self.data.mean())] * nsamp
        # cdata centers the data around its mean
        cdata = (self.data - pandas.DataFrame(means))
        # S estimates the covariance matrix
        S = sum(cdata[i:i + 1].transpose().dot(cdata[i:i + 1]) for i in range(nsamp))
        S /= nsamp                      # S has size dim * dim
        A0 = inv(S)                     # A0 has size dim * dim
        A1 = A0.dot(cdata.transpose())  # A1 has size dim * nsamp
        # Initializing A and B
        A = 0
        B = 0
        # Computing the sums in A and B

        for i in range(nsamp):
            row = cdata[i:i + 1]
            a = list(row.dot(A1)[0:1].values[0])
            A += sum(elt ** 3 for elt in a)
            B += a[i] ** 2

        # Normalization of A and B
        A /= (6 * nsamp)
        B /= nsamp
        B -= dim * (dim + 2)
        B *= sqrt(nsamp / (8 * dim * (dim + 2)))
        # A should follow a chi-2 distribution w/ chi_df degrees of freedom
        # B should follow a normal distribution
        chi_df = dim * (dim + 1) * (dim + 2) / 6
        pval_A = 1 - chi2.cdf(A, chi_df)
        pval_B = 1 - norm.cdf(B)
        A = A
        B = B
        return (A, B, pval_A, pval_B)


def doornik_hansen(data):
    """
    Perform the Doornik-Hansen test
    (https://doi.org/10.1111/j.1468-0084.2008.00537.x)

    This computes and transforms multivariate variants of the skewness
    and kurtosis, then computes a chi-square statistic on the results.
    """
    data = pandas.DataFrame(data)
    data = deepcopy(data)

    n = len(data)
    p = len(data.columns)
    # R is the correlation matrix, a scaling of the covariance matrix
    # R has dimensions dim * dim
    R = corrcoef(data.transpose())
    L, V = eigh(R)
    for i in range(p):
        if(L[i] <= 1e-12):
            L[i] = 0
        if(L[i] > 1e-12):
            L[i] = 1 / sqrt(L[i])
    L = diag(L)

    if(matrix_rank(R) < p):
        V = pandas.DataFrame(V)
        G = V.loc[:, (L != 0).any(axis=0)]
        data = data.dot(G)
        ppre = p
        p = data.size / len(data)
        raise ValueError("NOTE:Due that some eigenvalue resulted zero, a new data matrix was created. Initial number of variables = ", ppre, ", were reduced to = ", p)
        R = corrcoef(data.transpose())
        L, V = eigh(R)
        L = diag(L)

    means = [list(data.mean())] * n
    stddev = [list(data.std(ddof=0))] * n

    Z = (data - pandas.DataFrame(means)) / pandas.DataFrame(stddev)
    Zp = Z.dot(V)
    Zpp = Zp.dot(L)
    st = Zpp.dot(transpose(V))

    # skew is the multivariate skewness (dimension dim)
    # kurt is the multivariate kurtosis (dimension dim)
    skew = mean(power(st, 3), axis=0)
    kurt = mean(power(st, 4), axis=0)

    # Transform the skewness into a standard normal z1
    n2 = n * n
    b = 3 * (n2 + 27 * n - 70) * (n + 1) * (n + 3)
    b /= (n - 2) * (n + 5) * (n + 7) * (n + 9)
    w2 = -1 + sqrt(2 * (b - 1))
    d = 1 / sqrt(log(sqrt(w2)))
    y = skew * sqrt((w2 - 1) * (n + 1) * (n + 3) / (12 * (n - 2)))
    # Use numpy log/sqrt as math versions dont have array input
    z1 = d * nplog(y + npsqrt(y * y + 1))

    # Transform the kurtosis into a standard normal z2
    d = (n - 3) * (n + 1) * (n2 + 15 * n - 4)
    a = (n - 2) * (n + 5) * (n + 7) * (n2 + 27 * n - 70) / (6 * d)
    c = (n - 7) * (n + 5) * (n + 7) * (n2 + 2 * n - 5) / (6 * d)
    k = (n + 5) * (n + 7) * (n * n2 + 37 * n2 + 11 * n - 313) / (12 * d)
    al = a + (skew ** 2) * c
    chi = (kurt - 1 - (skew ** 2)) * k * 2
    z2 = (((chi / (2 * al)) ** (1 / 3)) - 1 + 1 / (9 * al)) * npsqrt(9 * al)
    kurt -= 3

    # omnibus normality statistic
    DH = z1.dot(z1.transpose()) + z2.dot(z2.transpose())
    AS = n / 6 * skew.dot(skew.transpose()) + n / 24 * kurt.dot(kurt.transpose())
    # degrees of freedom
    v = 2 * p
    # p-values
    PO = 1 - chi2.cdf(DH, v)
    PA = 1 - chi2.cdf(AS, v)

    return DH, AS, PO, PA


def diagcov(cov_mat, nsamples):
    """
    This test studies the population covariance matrix.
    Suppose it is of this form:
     ____________
    |     |     |
    |  1  |  3  |
    |_____|_____|
    |     |     |
    |     |  2  |
    |_____|_____|

    The test will first compute sums of elements on diagonals of 1, 2 or 3,
    and store them in the table diagsum of size 2 * dim:
    - First (dim / 2) lines = means of each diag. of 1 above leading diag.
    - Following (dim / 2) lines = means of each diag. of 2 above leading diag.
    - Following (dim / 2) lines = means of each diag. of 3 above leading diag.
    - Last (dim / 2) lines = means of each diag. of 3 below leading diag.

    We are making the assumption that each cell of the covariance matrix
    follows a normal distribution of variance 1 / n. Assuming independence
    of each cell in a diagonal, each diagonal sum of k elements should
    follow a normal distribution of variance k / n (hence of variance
    1 after normalization by n / k).

    We then compute the sum of the squares of all elements in diagnorm.
    If is supposed to look like a chi-square distribution
    """
    dim = len(cov_mat)
    n0 = dim // 2
    diagsum = [0] * (2 * dim)
    for i in range(1, n0):
        diagsum[i] = sum(cov_mat[j][i + j] for j in range(n0 - i))
        diagsum[i + n0] = sum(cov_mat[n0 + j][n0 + i + j] for j in range(n0 - i))
        diagsum[i + 2 * n0] = sum(cov_mat[j][n0 + i + j] for j in range(n0 - i))
        diagsum[i + 3 * n0] = sum(cov_mat[j][n0 - i + j] for j in range(n0 - i))
    # Diagnorm contains the normalized sums, which should be normal
    diagnorm = diagsum[:]
    for i in range(1, n0):
        nfactor = sqrt(nsamples / (n0 - i))
        diagnorm[i] *= nfactor
        diagnorm[i + n0] *= nfactor
        diagnorm[i + 2 * n0] *= nfactor
        diagnorm[i + 3 * n0] *= nfactor

    # Each diagnorm[i + _ * n0] should be a random normal variable
    chistat = sum(elt ** 2 for elt in diagnorm)
    pvalue = 1 - chi2.cdf(chistat, df=4 * (n0 - 1))
    return pvalue


def test_pysampler(nb_mu=100, nb_sig=100, nb_samp=100):
    """
    Test our Gaussian sampler on a bunch of samples.
    """
    print("Testing the sampler over Z with:\n")
    print("- {a} different centers\n".format(a=nb_mu))
    print("- {a} different sigmas\n".format(a=nb_sig))
    print("- {a} samples per center and sigma\n".format(a=nb_samp))
    assert(nb_samp >= 10 * chi2_bucket)
    q = 12289
    sig_min = 1.3
    sig_max = 1.8
    nb_rej = 0
    for i in range(nb_mu):
        mu = uniform(0, q)
        for j in range(nb_sig):
            sigma = uniform(sig_min, sig_max)
            list_samples = [samplerz(mu, sigma) for _ in range(nb_samp)]
            v = UnivariateSamples(mu, sigma, list_samples)
            if (v.chi2_pvalue < pmin):
                nb_rej += 1
    print("The test failed {k} times out of {t} (expected {e})".format(k=nb_rej, t=nb_mu * nb_sig, e=round(pmin * nb_mu * nb_sig)))


def parse_univariate_file(filename):
    """
    Parse a file containing several univariate samples.

    Input:
    - the file name of a file containing 2 * k lines of this form:
      - line 2 * i: "mu = xxx, sigma = yyy"
      - line 2 * i + 1: zzz samples

    Output:
    - a Python list containing k elements:
      - each element is of the form (mu, sigma, data):
        - mu = expected center of a Gaussian
        - sigma = expected standard deviation of a Gaussian
        - data = samples from a distribution to test against the Gaussian
                 parametrized by (mu, sigma)
    """
    # Initialize the output as the empty list
    data_list = []
    # Open the file
    with open(filename) as f:
        while True:
            line1 = f.readline()
            line2 = f.readline()
            if not line2:
                break  # EOF
            # Parsing mu and sigma
            (_, mu, sigma, _) = re.split("mu = |, sigma = |\n", line1)
            mu = float(mu)
            sigma = float(sigma)
            # Parsing the samples z
            data = re.split(", |,\n", line2)
            data = [int(elt) for elt in data[:-1]]
            # Add the triple (mu, sigma, data) to the output
            data_list += [(mu, sigma, data)]
    return data_list


def parse_multivariate_file(filename):
    """
    Parse a file containing several multivariate samples.

    Input:
    - the file name of a file containing k lines
      - each line corresponds to a multivariate sample
      - the samples are all assumed to be from the same distribution

    Output:
    - sigma: the expected standard deviation of the samples
    - data: a Python list of length k, containing all the samples
    """
    with open(filename) as f:
        sigma = 0
        data = []
        while True:
            # Parse each line
            line = f.readline()
            if not line:
                break  # EOF
            sample = re.split(", |,\n", line)
            sample = [int(elt) for elt in sample[:-1]]
            data += [sample]
            sigma += sum(elt ** 2 for elt in sample)
        # sigma is the expected sigma based on the samples
        sigma = sqrt(sigma / (len(data) * len(data[0])))
    return (sigma, data)


def test_falcon():
    """
    We test samples obtained directly from Falcon's reference implementation.
    We test:
    - univariate samples from the sampler over Z
    - multivariate samples from the signature scheme
    """

    # We first test the Gaussian sampler over Z, using the samples in:
    # - testdata/sampler_fpnative
    # - testdata/sampler_avx2
    # - testdata/sampler_fpemu
    # Each file should be formatted as to be parsable with parse_univariate_file()
    for filename in ["sampler_fpnative", "sampler_avx2", "sampler_fpemu"]:
        print("Testing data in file testdata/{file}:".format(file=filename))
        data_list = parse_univariate_file("testdata/" + filename)
        # n_mu_and_sig is the number of different couples (mu, sigma)
        n_mu_and_sig = len(data_list)
        n_invalid = 0
        for elt in data_list:
            (mu, sigma, data) = elt
            u = UnivariateSamples(mu, sigma, data)
            n_invalid += (u.is_valid is False)
        print("- We tested {k} different (mu, sigma) list of samples".format(k=n_mu_and_sig))
        print("- We found {k} invalid list of samples\n".format(k=n_invalid))

    # Now we test the distribution of signatures using the multivariate test.
    # Each element of this filelist is a text file containing a large number
    # of multivariate samples. Each file should be formatted as to be
    # parsable with parse_multivariate_file()
    filelist = [
        "falcon64_avx2",
        "falcon128_avx2",
        "falcon256_avx2",
        "falcon512_avx2",
        "falcon1024_avx2",
        "falcon64_fpnative",
        "falcon128_fpnative",
        "falcon256_fpnative",
        "falcon512_fpnative",
        "falcon1024_fpnative",
        "falcon64_fpemu_big",
        "falcon128_fpemu_big",
        "falcon256_fpemu_big",
        "falcon512_fpemu_big",
        "falcon1024_fpemu_big",
    ]
    for filename in filelist:
        print("\n\nTesting data in file testdata/{file}:".format(file=filename))
        (sigma, data) = parse_multivariate_file("testdata/" + filename)
        mv = MultivariateSamples(sigma, data)
        print(mv)
    return


def test_sig(n=128, nb_sig=1000, perturb=False, level=0):
    """
    Test signatures output by a Python implementation of Falcon.
    This test allow to perturb the FFT by setting the rightmost node
    of the FFT tree (of the private key) to 0. One can check that, at
    least for moderate levels (0 to 4), the test will end up detecting
    (via diagcov) that the signatures output do not follow the correct
    distribution.

    Input:
    - n: the degree of the ring
    - nb_sig: number of signatures
    - perturb: if set to 1, one node in the FFT tree is set to 0
    - level: determines which node (the rightmost one at a given level)
      is set to 0
    """
    start = time.time()
    # Generate a private key
    sk = falcon.SecretKey(n)
    # Perturb the FFT tree
    if perturb is True:
        # Check that the level is less than the FFT tree depth
        assert(1 << level) < n
        u, k = sk.T_fft, n
        # Find the node
        for _ in range(level):
            u = u[2]
            k >>= 1
        # Zero-ize the node
        u[0] = [0] * k
    end = time.time()
    print("Took {t:.2f} seconds to generate the private key.".format(t=end - start))

    # Compute signatures
    message = "0"
    start = time.time()
    list_signatures = [sk.sign(message, reject=False) for _ in range(nb_sig)]
    # Strip away the nonces and concatenate the s_1's and s_2's
    list_signatures = [sig[1][0] + sig[1][1] for sig in list_signatures]
    end = time.time()
    print("Took {t:.2f} seconds to generate the samples.".format(t=end - start))
    # Perform the statistical test
    start = time.time()
    samples_data = MultivariateSamples(sk.sigma, list_signatures)
    end = time.time()
    print("Took {t:.2f} seconds to run a statistical test.".format(t=end - start))
    return sk, samples_data


def test_rejind(mu, sigma):
    """
    input assumes dataset with num rejs (a) for each output (b)
    to form the data structure [(a,b)]*n to test for independence
    """

    # parameters to generate data
    n = 10000
    mu = 0
    nb_mu = 100
    sigma = 1.5
    q = 12289

    # assumed data input for testing:
    # output given as a tuple (x,#reps)
    data = [samplerz_rep(mu, sigma) for _ in range(n)]

    counter = Counter(map(tuple,data))
    values, rejects = zip(*data)
    results = []
    mu = 0
    for i in range(nb_mu):
        list_samples = [samplerz_rep(mu, sigma) for _ in range(n)]
        counter = dict(Counter(map(tuple, list_samples)))
        result = defaultdict(int)
        for key in sorted(counter.keys()):
            result[key[1]] += int(counter[key])
        result = dict(result)
        results.append(result)
    mu += q / nb_mu

    # sort data
    df = pandas.DataFrame(results)
    df = df.fillna(0)
    df = df.sort_index(axis=1)
    print(df)

    # plot
    plt.figure(figsize=(24, 5))
    plt.pcolor(df)
    plt.colorbar
    plt.yticks(arange(0, len(df.index), step=10), fontsize=17)
    plt.xticks(arange(0.5, len(df.columns), 1), df.columns, fontsize=17)

    plt.rcParams["axes.grid"] = False
    plt.xlim((0, 9))
    plt.xlabel('Number of Rejections', fontsize=21)
    plt.ylabel('Dataset Number', fontsize=21)
    plt.savefig('rejections.eps', format='eps', bbox_inches="tight", pad_inches=0)
    plt.show()


def test_basesampler(mu, sigma):
    """
    A set of visual tests, assuming you have failed some tests,
    either for univariate data input or generated below.
    """

    # generate data
    n = 100000
    mu = 0
    sigma = 1.5
    data = [samplerz(mu, sigma) for _ in range(n)]

    # histogram
    hist, bins = histogram(data, bins=abs(min(data)) + max(data))
    x = bins[:-1]
    y1 = hist
    y2 = norm.pdf(x, mu, sigma) * n
    plt.bar(x, y1, width=1.25, color='blue', edgecolor='none', label='Gauss Samples')
    plt.plot(x, y2, '-r', label='Gauss Expected')
    plt.xlabel('$x$')
    plt.ylabel('pdf$(x)$')
    plt.legend(loc='upper right')
    plt.title("Gaussian Samples, Observed vs Expected", fontweight="bold", fontsize=12)
    plt.savefig('histogram.eps', format='eps', bbox_inches="tight", pad_inches=0)

    # qqplot
    r2 = stats.linregress(sort(hist), sort(y2))[2] ** 2
    plt.title('R-Squared = %0.20f' % r2, fontsize=9)
    plt.suptitle("QQ plot for Univariate Normality of Gaussian Samples", fontweight="bold", fontsize=12)
    plt.savefig('qqplot_test.eps', format='eps', bbox_inches="tight", pad_inches=0)
    plt.show()


#######################
# Supplementary Stuff #
#######################
def estimate_sphericity(covar_matrix):
    """
    Given a sample covariance matrix (in pandas DataFrame format), compute
    the box-index and centered box-index of the matrix.

    Both values should tend to 1 if the sample covariance matrix is the identity matrix
    """
    covariance = array(covar_matrix)
    dim = len(covariance)
    # Compute an estimator of the population covariance matrix
    # from the sample covariance matrix
    cov_mean = array([row.mean() for row in covariance])
    cov_meanmean = cov_mean.mean()
    cov_centered = deepcopy(covariance)
    for i in range(dim):
        for j in range(dim):
            cov_centered[i][j] += cov_meanmean - cov_mean[i] - cov_mean[j]
    # Compute the (centered) Box index
    box_num = sum(covariance[i][i] for i in range(dim)) ** 2
    box_den = (dim - 1) * sum(covariance[i][j] ** 2 for i in range(dim) for j in range(dim))
    box_index = box_num / box_den
    cbox_num = sum(cov_centered[i][i] for i in range(dim)) ** 2
    cbox_den = (dim - 1) * sum(cov_centered[i][j] ** 2 for i in range(dim) for j in range(dim))
    cbox_index = cbox_num / cbox_den
    # Compute eigenvalues (the eigenvalues of the centered and
    # non-centered covariance matrices seem to be the same!)
    eigen = eig(cov_centered)[0]
    # Compute V
    V = (sum(elt for elt in eigen) ** 2) / sum(elt ** 2 for elt in eigen)
    statistic = ((dim - 1) ** 2) * (V - 1 / (dim - 1)) / (2 * dim)
    df = (dim * (dim - 1) / 2) - 1
    print("statistic      = {s}".format(s=statistic))
    print("deg of freedom = {d}".format(d=df))
    return box_index, cbox_index


def qqplot(data):
    """
    https://www.itl.nist.gov/div898/handbook/eda/section3/qqplot.htm
    """
    data = pandas.DataFrame(data)
    data = deepcopy(data)

    S = cov(data.transpose(), bias=1)
    n = len(data)
    p = len(data.columns)

    means = [list(data.mean())] * n
    difT = data - pandas.DataFrame(means)
    Dj = diag(difT.dot(inv(S)).dot(difT.transpose()))
    Y = data.dot(inv(S)).dot(data.transpose())
    Ytdiag = array(pandas.DataFrame(diag(Y.transpose())))
    Djk = - 2 * Y.transpose()
    Djk += tile(Ytdiag, (1, n)).transpose()
    Djk += tile(Ytdiag, (1, n))
    Djk_quick = []
    for i in range(n):
        Djk_quick += list(Djk.values[i])

    chi2_random = chi2.rvs(p - 1, size=len(Dj))
    chi2_random = sort(chi2_random)
    r2 = stats.linregress(sort(Dj), sort(chi2_random))[2] ** 2
    plt.title('R-Squared = %0.20f' % r2, fontsize=9)
    plt.suptitle("QQ plot for Multivariate Normality", fontweight="bold", fontsize=12)

    plt.savefig('qqplot.eps', format='eps', bbox_inches="tight", pad_inches=0)
    plt.show()


def csv_testing():
    """
    Basic script for importing and testing csv files
    """
    with open('gaussian_samples.csv', "r") as csvfile:
        data = []
        for row in csv.reader(csvfile, delimiter=','):
            data.append(int(row[0]))
        test_basesampler(0,1.5,data)
