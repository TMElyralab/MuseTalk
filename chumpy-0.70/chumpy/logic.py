"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__author__ = 'matt'


__all__ = [] # added to incrementally below

from . import ch
from .ch import Ch
import numpy as np

class LogicFunc(Ch):
    dterms = 'a' # we keep this here so that changes to children of "a" will trigger cache changes
    terms = 'args', 'kwargs', 'funcname'

    def compute_r(self):
        arr = self.a
        fn = getattr(np, self.funcname)
        return fn(arr, *self.args, **self.kwargs)

    def compute_dr_wrt(self, wrt):
        pass


unaries = 'all', 'any', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'logical_not'
for unary in unaries:
    exec("def %s(a, *args, **kwargs): return LogicFunc(a=a, args=args, kwargs=kwargs, funcname='%s')" % (unary, unary))
__all__ += unaries



if __name__ == '__main__':
    from . import ch
    print(all(np.array([1,2,3])))
    print(isinf(np.array([0,2,3])))
