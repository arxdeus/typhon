
import internal.static as static

from .. import *

from .pcg64 import PCG64, PCG64DXSM
from .philox import Philox
from .mt19937 import MT19937
from .sfc64 import SFC64
from .bitgen import BitGenerator, Generator
from .seed import SeedSequence
from .legacy import RandomState, _rand

def default_rng(seed = None):
    if isinstance(seed, BitGenerator):
        return Generator(seed)

    if isinstance(seed, Generator):
        return seed

    return Generator(PCG64(seed))

# Module level legacy API: methods of the hidden global RandomState, as in
# numpy's mtrand (np.random.seed(s) then np.random.randn(...) etc. reproduce
# numpy bit for bit). default_rng()/Generator is the new API.

def seed(seed = None):
    _rand.seed(seed)

def get_bit_generator():
    return _rand._bitgen.state

def beta(*args, **kwargs):
    return _rand.beta(*args, **kwargs)

def binomial(*args, **kwargs):
    return _rand.binomial(*args, **kwargs)

def bytes(*args, **kwargs):
    return _rand.bytes(*args, **kwargs)

def chisquare(*args, **kwargs):
    return _rand.chisquare(*args, **kwargs)

def choice(*args, **kwargs):
    return _rand.choice(*args, **kwargs)

def dirichlet(*args, **kwargs):
    return _rand.dirichlet(*args, **kwargs)

def exponential(*args, **kwargs):
    return _rand.exponential(*args, **kwargs)

def f(*args, **kwargs):
    return _rand.f(*args, **kwargs)

def gamma(*args, **kwargs):
    return _rand.gamma(*args, **kwargs)

def geometric(*args, **kwargs):
    return _rand.geometric(*args, **kwargs)

def get_state(*args, **kwargs):
    return _rand.get_state(*args, **kwargs)

def gumbel(*args, **kwargs):
    return _rand.gumbel(*args, **kwargs)

def hypergeometric(*args, **kwargs):
    return _rand.hypergeometric(*args, **kwargs)

def laplace(*args, **kwargs):
    return _rand.laplace(*args, **kwargs)

def logistic(*args, **kwargs):
    return _rand.logistic(*args, **kwargs)

def lognormal(*args, **kwargs):
    return _rand.lognormal(*args, **kwargs)

def logseries(*args, **kwargs):
    return _rand.logseries(*args, **kwargs)

def multinomial(*args, **kwargs):
    return _rand.multinomial(*args, **kwargs)

def multivariate_normal(*args, **kwargs):
    return _rand.multivariate_normal(*args, **kwargs)

def negative_binomial(*args, **kwargs):
    return _rand.negative_binomial(*args, **kwargs)

def noncentral_chisquare(*args, **kwargs):
    return _rand.noncentral_chisquare(*args, **kwargs)

def noncentral_f(*args, **kwargs):
    return _rand.noncentral_f(*args, **kwargs)

def normal(*args, **kwargs):
    return _rand.normal(*args, **kwargs)

def pareto(*args, **kwargs):
    return _rand.pareto(*args, **kwargs)

def permutation(*args, **kwargs):
    return _rand.permutation(*args, **kwargs)

def poisson(*args, **kwargs):
    return _rand.poisson(*args, **kwargs)

def power(*args, **kwargs):
    return _rand.power(*args, **kwargs)

def rand(*args, **kwargs):
    return _rand.rand(*args, **kwargs)

def randint(*args, **kwargs):
    return _rand.randint(*args, **kwargs)

def randn(*args, **kwargs):
    return _rand.randn(*args, **kwargs)

def random(*args, **kwargs):
    return _rand.random(*args, **kwargs)

def random_integers(*args, **kwargs):
    return _rand.random_integers(*args, **kwargs)

def random_sample(*args, **kwargs):
    return _rand.random_sample(*args, **kwargs)

def ranf(*args, **kwargs):
    return _rand.ranf(*args, **kwargs)

def rayleigh(*args, **kwargs):
    return _rand.rayleigh(*args, **kwargs)

def sample(*args, **kwargs):
    return _rand.sample(*args, **kwargs)

def set_state(*args, **kwargs):
    return _rand.set_state(*args, **kwargs)

def shuffle(*args, **kwargs):
    return _rand.shuffle(*args, **kwargs)

def standard_cauchy(*args, **kwargs):
    return _rand.standard_cauchy(*args, **kwargs)

def standard_exponential(*args, **kwargs):
    return _rand.standard_exponential(*args, **kwargs)

def standard_gamma(*args, **kwargs):
    return _rand.standard_gamma(*args, **kwargs)

def standard_normal(*args, **kwargs):
    return _rand.standard_normal(*args, **kwargs)

def standard_t(*args, **kwargs):
    return _rand.standard_t(*args, **kwargs)

def tomaxint(*args, **kwargs):
    return _rand.tomaxint(*args, **kwargs)

def triangular(*args, **kwargs):
    return _rand.triangular(*args, **kwargs)

def uniform(*args, **kwargs):
    return _rand.uniform(*args, **kwargs)

def vonmises(*args, **kwargs):
    return _rand.vonmises(*args, **kwargs)

def wald(*args, **kwargs):
    return _rand.wald(*args, **kwargs)

def weibull(*args, **kwargs):
    return _rand.weibull(*args, **kwargs)

def zipf(*args, **kwargs):
    return _rand.zipf(*args, **kwargs)
