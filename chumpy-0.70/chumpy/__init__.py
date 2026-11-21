from .ch import *
from .logic import *

from .optimization import minimize
from . import extras
from . import testing
from .version import version as __version__

from .version import version as __version__

from numpy import bool, int, float, complex, object, unicode, str, nan, inf

def test():
    from os.path import split
    import unittest
    test_loader= unittest.TestLoader()
    test_loader = test_loader.discover(split(__file__)[0])
    test_runner = unittest.TextTestRunner()
    test_runner.run( test_loader )


demos = {}

demos['scalar'] = """
import chumpy as ch

[x1, x2, x3] = ch.array(10), ch.array(20), ch.array(30)
result = x1+x2+x3
print result # prints [ 60.]
print result.dr_wrt(x1) # prints 1
"""

demos['show_tree'] = """
import chumpy as ch

[x1, x2, x3] = ch.array(10), ch.array(20), ch.array(30)
for i in range(3): x2 = x1 + x2 + x3

x2.dr_wrt(x1) # pull cache 
x2.dr_wrt(x3) # pull cache
x1.label='x1' # for clarity in show_tree()
x2.label='x2' # for clarity in show_tree()
x3.label='x3' # for clarity in show_tree()
x2.show_tree(cachelim=1e-4) # in MB
"""

demos['matrix'] = """
import chumpy as ch

x1, x2, x3, x4 = ch.eye(10), ch.array(1), ch.array(5), ch.array(10)
y = x1*(x2-x3)+x4
print y
print y.dr_wrt(x2)
"""

demos['linalg'] = """
import chumpy as ch

m = [ch.random.randn(100).reshape((10,10)) for i in range(3)]
y = m[0].dot(m[1]).dot(ch.linalg.inv(m[2])) * ch.linalg.det(m[0])
print y.shape
print y.dr_wrt(m[0]).shape
"""

demos['inheritance'] = """
import chumpy as ch
import numpy as np

class Sin(ch.Ch):

    dterms = ('x',)

    def compute_r(self):
        return np.sin(self.x.r)

    def compute_dr_wrt(self, wrt):
        import scipy.sparse
        if wrt is self.x:
            result = np.cos(self.x.r)
            return scipy.sparse.diags([result.ravel()], [0]) if len(result)>1 else np.atleast_2d(result)

x1 = Ch([10,20,30])
result = Sin(x1) # or "result = Sin(x=x1)"
print result.r
print result.dr_wrt(x1)
"""

demos['optimization'] = """
import chumpy as ch

x = ch.zeros(10)
y = ch.zeros(10)

# Beale's function
e1 = 1.5 - x + x*y
e2 = 2.25 - x  + x*(y**2)
e3 = 2.625 - x + x*(y**3)

objective = {'e1': e1, 'e2': e2, 'e3': e3}
ch.minimize(objective, x0=[x,y], method='dogleg')
print x # should be all 3.0
print y # should be all 0.5
"""




def demo(which=None):
    if which not in demos:
        print('Please indicate which demo you want, as follows:')
        for key in demos:
            print("\tdemo('%s')" % (key,))
        return

    print('- - - - - - - - - - - <CODE> - - - - - - - - - - - -')
    print(demos[which])
    print('- - - - - - - - - - - </CODE> - - - - - - - - - - - -\n')
    exec('global np\n' + demos[which], globals(), locals())
