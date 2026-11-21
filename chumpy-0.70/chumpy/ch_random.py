"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import numpy.random
from .ch import Ch

api_not_implemented = ['choice','bytes','shuffle','permutation']

api_wrapped_simple = [
    # simple random data
    'rand','randn','randint','random_integers','random_sample','random','ranf','sample',
    
    # distributions
    'beta','binomial','chisquare','dirichlet','exponential','f','gamma','geometric','gumbel','hypergeometric',
    'laplace','logistic','lognormal','logseries','multinomial','multivariate_normal','negative_binomial',
    'noncentral_chisquare','noncentral_f','normal','pareto','poisson','power','rayleigh','standard_cauchy',
    'standard_exponential','standard_gamma','standard_normal','standard_t','triangular','uniform','vonmises',
    'wald','weibull','zipf']
    
api_wrapped_direct = ['seed', 'get_state', 'set_state']

for rtn in api_wrapped_simple:
    exec('def %s(*args, **kwargs) : return Ch(numpy.random.%s(*args, **kwargs))' % (rtn, rtn))

for rtn in api_wrapped_direct:
    exec('%s = numpy.random.%s' % (rtn, rtn))
    
__all__ = api_wrapped_simple + api_wrapped_direct

