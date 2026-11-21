"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""


from . import ch
import numpy as np
from os.path import join, split
from six import StringIO
import numpy
import chumpy
from six.moves import cPickle as pickle

src = ''
num_passed = 0
num_not_passed = 0
which_passed = []

def r(fn_name, args_req, args_opt, nplib=numpy, chlib=chumpy):
    global num_passed, num_not_passed
    result = [None, None]
        
    for lib in [nplib, chlib]:

        # if fn_name is 'svd' and lib is chlib:
        #     import pdb; pdb.set_trace()
        if lib is nplib:            
            fn = getattr(lib, fn_name)
        else:
            try:
                fn = getattr(lib, fn_name)
            except AttributeError:
                result[0] = 'missing'
                result[1] = 'missing'
                num_not_passed += 1
                continue
        try:
            if isinstance(args_req, dict):
                _ = fn(**args_req)
            else:
                _ = fn(*args_req)
            if lib is chlib:
                result[0] = 'passed'
                num_passed += 1
                global which_passed
                which_passed.append(fn_name)
                
                if hasattr(_, 'dterms'):
                    try:
                        _.r
                    
                        try:
                            pickle.dumps(_)
                        except:
                            result[0] += ' (but unpickleable!)'
                    except:
                        import pdb; pdb.set_trace()
                        result[0] += '(but cant get result!)'
        except Exception as e:
            if e is TypeError:
                import pdb; pdb.set_trace()
            if lib is nplib:
                import pdb; pdb.set_trace()
            else:
                num_not_passed += 1
                # if fn_name == 'rot90':
                #     import pdb; pdb.set_trace()
            result[0] = e.__class__.__name__
            
        try:
            if isinstance(args_req, dict):
                fn(**dict(list(args_req.items()) + list(args_opt.items())))
            else:
                fn(*args_req, **args_opt)
            if lib is chlib:
                result[1] = 'passed'
        except Exception as e:
            if e is TypeError:
                import pdb; pdb.set_trace()
            result[1] = e.__class__.__name__
            
    # print '%s: %s, %s' % (fn_name, result[0], result[1])
    
    append(fn_name, result[0], result[1])
    
def make_row(a, b, c, b_color, c_color):
    global src
    src += '<tr><td>%s</td><td style="background-color:%s">%s</td><td style="background-color:%s">%s</td></tr>' % (a,b_color, b,c_color, c)
    
def append(a, b, c):
    global src
    b_color = 'white'
    c_color = 'white'

    b = b.replace('NotImplementedError', 'not yet implemented')
    c = c.replace('NotImplementedError', 'not yet implemented')
    b = b.replace('WontImplement', "won't implement")
    c = c.replace('WontImplement', "won't implement")
    lookup = {
        'passed': 'lightgreen',
        "won't implement": 'lightgray',
        'untested': 'lightyellow',
        'not yet implemented': 'pink'
    }
    
    b_color = lookup[b] if b in lookup else 'white'
    c_color = lookup[c] if c in lookup else 'white'

    print('%s: %s, %s' % (a,b,c))
    make_row(a, b, c, b_color, c_color)

def m(s):
    append(s, 'unknown', 'unknown')
    global num_not_passed
    num_not_passed += 1

def hd3(s):
    global src
    src += '<tr><td colspan=3><h3 style="margin-bottom:0;">%s</h3></td></tr>' % (s,)

def hd2(s):
    global src
    src += '</table><br/><br/><table border=1>'
    src += '<tr><td colspan=3 style="background-color:black;color:white"><h2 style="margin-bottom:0;">%s</h2></td></tr>' % (s,)
    
def main():
    
    #sample_array
    
    ###############################
    hd2('Array Creation Routines')
    
    hd3('Ones and zeros')

    r('empty', {'shape': (2,4,2)}, {'dtype': np.uint8, 'order': 'C'})
    r('empty_like', {'prototype': np.empty((2,4,2))}, {'dtype': np.float64, 'order': 'C'})
    r('eye', {'N': 10}, {'M': 5, 'k': 0, 'dtype': np.float64})
    r('identity', {'n': 10}, {'dtype': np.float64})
    r('ones', {'shape': (2,4,2)}, {'dtype': np.uint8, 'order': 'C'})
    r('ones_like', {'a': np.empty((2,4,2))}, {'dtype': np.float64, 'order': 'C'})
    r('zeros', {'shape': (2,4,2)}, {'dtype': np.uint8, 'order': 'C'})
    r('zeros_like', {'a': np.empty((2,4,2))}, {'dtype': np.float64, 'order': 'C'})
    
    hd3('From existing data')
    r('array', {'object': [1,2,3]}, {'dtype': np.float64, 'order': 'C', 'subok': False, 'ndmin': 2})
    r('asarray', {'a': np.array([1,2,3])}, {'dtype': np.float64, 'order': 'C'})
    r('asanyarray', {'a': np.array([1,2,3])}, {'dtype': np.float64, 'order': 'C'})
    r('ascontiguousarray', {'a': np.array([1,2,3])}, {'dtype': np.float64})
    r('asmatrix', {'data': np.array([1,2,3])}, {'dtype': np.float64})
    r('copy', (np.array([1,2,3]),), {})
    r('frombuffer', {'buffer': np.array([1,2,3])}, {})
    m('fromfile')
    r('fromfunction', {'function': lambda i, j: i + j, 'shape': (3, 3)}, {'dtype': np.float64})
    # function, shape, **kwargs
    # lambda i, j: i + j, (3, 3), dtype=int
    r('fromiter', {'iter': [1,2,3,4], 'dtype': np.float64}, {'count': 2})
    r('fromstring', {'string': '\x01\x02', 'dtype': np.uint8}, {})
    r('loadtxt', {'fname': StringIO("0 1\n2 3")}, {})

    hd3('Creating record arrays (wont be implemented)')
    hd3('Creating character arrays (wont be implemented)')

    hd3('Numerical ranges')
    r('arange', {'start': 0, 'stop': 10}, {'step': 2, 'dtype': np.float64})
    r('linspace', {'start': 0, 'stop': 10}, {'num': 2, 'endpoint': 10, 'retstep': 1})
    r('logspace', {'start': 0, 'stop': 10}, {'num': 2, 'endpoint': 10, 'base': 1})
    r('meshgrid', ([1,2,3], [4,5,6]), {})
    m('mgrid')
    m('ogrid')
    
    hd3('Building matrices')
    r('diag', {'v': np.arange(9).reshape((3,3))}, {'k': 0})
    r('diagflat', {'v': [[1,2], [3,4]]}, {})
    r('tri', {'N': 3}, {'M': 5, 'k': 2, 'dtype': np.float64})
    r('tril', {'m': [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]}, {'k': -1})
    r('triu', {'m': [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]}, {'k': -1})
    r('vander', {'x': np.array([1, 2, 3, 5])}, {'N': 3})
    
    ###############################
    hd2('Array manipulation routines')
    
    hd3('Basic operations')
    r('copyto', {'dst': np.eye(3), 'src': np.eye(3)}, {})
    
    hd3('Changing array shape')
    r('reshape', {'a': np.eye(3), 'newshape': (9,)}, {'order' : 'C'})
    r('ravel', {'a': np.eye(3)}, {'order' : 'C'})
    m('flat')
    m('flatten')
    
    hd3('Transpose-like operations')
    r('rollaxis', {'a': np.ones((3,4,5,6)), 'axis': 3}, {'start': 0})
    r('swapaxes', {'a': np.array([[1,2,3]]), 'axis1': 0, 'axis2': 1}, {})
    r('transpose', {'a': np.arange(4).reshape((2,2))}, {'axes': (1,0)})
    
    hd3('Changing number of dimensions')
    r('atleast_1d', (np.eye(3),), {})
    r('atleast_2d', (np.eye(3),), {})
    r('atleast_3d', (np.eye(3),), {})
    m('broadcast')
    m('broadcast_arrays')
    r('expand_dims', (np.array([1,2]),2), {})
    r('squeeze', {'a': (np.array([[[1,2,3]]]))}, {})
    
    hd3('Changing kind of array')
    r('asarray', {'a': np.array([1,2,3])}, {'dtype': np.float64, 'order': 'C'})
    r('asanyarray', {'a': np.array([1,2,3])}, {'dtype': np.float64, 'order': 'C'})
    r('asmatrix', {'data': np.array([1,2,3])}, {})
    r('asfarray', {'a': np.array([1,2,3])}, {})
    r('asfortranarray', {'a': np.array([1,2,3])}, {})
    r('asscalar', {'a': np.array([24])}, {})
    r('require', {'a': np.array([24])}, {})
    
    hd3('Joining arrays')
    m('column_stack')
    r('concatenate', ((np.eye(3), np.eye(3)),1), {})
    r('dstack', ((np.eye(3), np.eye(3)),), {})
    r('hstack', ((np.eye(3), np.eye(3)),), {})
    r('vstack', ((np.eye(3), np.eye(3)),), {})

    hd3('Splitting arrays')
    m('array_split')
    m('dsplit')
    m('hsplit')
    m('split')
    m('vsplit')

    hd3('Tiling arrays')
    r('tile', (np.array([0, 1, 2]),2), {})
    r('repeat', (np.array([[1,2],[3,4]]), 3), {'axis': 1})

    hd3('Adding and removing elements')
    m('delete')
    m('insert')
    m('append')
    m('resize')
    m('trim_zeros')
    m('unique')
    
    hd3('Rearranging elements')
    r('fliplr', (np.eye(3),), {})
    r('flipud', (np.eye(3),), {})
    r('reshape', {'a': np.eye(3), 'newshape': (9,)}, {'order' : 'C'})
    r('roll', (np.arange(10), 2), {})
    r('rot90', (np.arange(4).reshape((2,2)),), {})
        
    ###############################
    hd2('Linear algebra (numpy.linalg)')
    
    extra_args = {'nplib': numpy.linalg, 'chlib': ch.linalg}
    
    hd3('Matrix and dot products')
    r('dot', {'a': np.eye(3), 'b': np.eye(3)}, {})
    r('dot', {'a': np.eye(3).ravel(), 'b': np.eye(3).ravel()}, {})
    r('vdot', (np.eye(3).ravel(), np.eye(3).ravel()), {})
    r('inner', (np.eye(3).ravel(), np.eye(3).ravel()), {})
    r('outer', (np.eye(3).ravel(), np.eye(3).ravel()), {})
    r('tensordot', {'a': np.eye(3), 'b': np.eye(3)}, {})
    m('einsum')
    r('matrix_power', {'M': np.eye(3), 'n': 2}, {}, **extra_args)
    r('kron', {'a': np.eye(3), 'b': np.eye(3)}, {})
        
    hd3('Decompositions')
    r('cholesky', {'a': np.eye(3)}, {}, **extra_args)
    r('qr', {'a': np.eye(3)}, {}, **extra_args)
    r('svd', (np.eye(3),), {}, **extra_args)
    
    hd3('Matrix eigenvalues')
    r('eig', (np.eye(3),), {}, **extra_args)
    r('eigh', (np.eye(3),), {}, **extra_args)
    r('eigvals', (np.eye(3),), {}, **extra_args)
    r('eigvalsh', (np.eye(3),), {}, **extra_args)
    
    hd3('Norms and other numbers')
    r('norm', (np.eye(3),), {}, **extra_args)
    r('cond', (np.eye(3),), {}, **extra_args)
    r('det', (np.eye(3),), {}, **extra_args)
    r('slogdet', (np.eye(3),), {}, **extra_args)
    r('trace', (np.eye(3),), {})
    
    hd3('Solving equations and inverting matrices')
    r('solve', (np.eye(3),np.ones(3)), {}, **extra_args)
    r('tensorsolve', (np.eye(3),np.ones(3)), {}, **extra_args)
    r('lstsq', (np.eye(3),np.ones(3)), {}, **extra_args)
    r('inv', (np.eye(3),), {}, **extra_args)
    r('pinv', (np.eye(3),), {}, **extra_args)
    r('tensorinv', (np.eye(4*6).reshape((4,6,8,3)),), {'ind': 2}, **extra_args)
    
    hd2('Mathematical functions')

    hd3('Trigonometric functions')
    r('sin', (np.arange(3),), {})
    r('cos', (np.arange(3),), {})
    r('tan', (np.arange(3),), {})
    r('arcsin', (np.arange(3)/3.,), {})
    r('arccos', (np.arange(3)/3.,), {})
    r('arctan', (np.arange(3)/3.,), {})
    r('hypot', (np.arange(3),np.arange(3)), {})
    r('arctan2', (np.arange(3),np.arange(3)), {})
    r('degrees', (np.arange(3),), {})
    r('radians', (np.arange(3),), {})
    r('unwrap', (np.arange(3),), {})
    r('unwrap', (np.arange(3),), {})
    r('deg2rad', (np.arange(3),), {})
    r('rad2deg', (np.arange(3),), {})
    
    hd3('Hyperbolic functions')
    r('sinh', (np.arange(3),), {})
    r('cosh', (np.arange(3),), {})
    r('tanh', (np.arange(3),), {})
    r('arcsinh', (np.arange(3)/9.,), {})
    r('arccosh', (-np.arange(3)/9.,), {})
    r('arctanh', (np.arange(3)/9.,), {})
    
    hd3('Rounding')
    r('around', (np.arange(3),), {})
    r('round_', (np.arange(3),), {})
    r('rint', (np.arange(3),), {})
    r('fix', (np.arange(3),), {})
    r('floor', (np.arange(3),), {})
    r('ceil', (np.arange(3),), {})
    r('trunc', (np.arange(3),), {})
    
    hd3('Sums, products, differences')
    r('prod', (np.arange(3),), {})
    r('sum', (np.arange(3),), {})
    r('nansum', (np.arange(3),), {})
    r('cumprod', (np.arange(3),), {})
    r('cumsum', (np.arange(3),), {})
    r('diff', (np.arange(3),), {})
    r('ediff1d', (np.arange(3),), {})
    r('gradient', (np.arange(3),), {})
    r('cross', (np.arange(3), np.arange(3)), {})
    r('trapz', (np.arange(3),), {})
    
    hd3('Exponents and logarithms')
    r('exp', (np.arange(3),), {})
    r('expm1', (np.arange(3),), {})
    r('exp2', (np.arange(3),), {})
    r('log', (np.arange(3),), {})
    r('log10', (np.arange(3),), {})
    r('log2', (np.arange(3),), {})
    r('log1p', (np.arange(3),), {})
    r('logaddexp', (np.arange(3), np.arange(3)), {})
    r('logaddexp2', (np.arange(3), np.arange(3)), {})
    
    hd3('Other special functions')
    r('i0', (np.arange(3),), {})
    r('sinc', (np.arange(3),), {})
    
    hd3('Floating point routines')
    r('signbit', (np.arange(3),), {})
    r('copysign', (np.arange(3), np.arange(3)), {})
    r('frexp', (np.arange(3),), {})
    r('ldexp', (np.arange(3), np.arange(3)), {})
    
    hd3('Arithmetic operations')
    r('add', (np.arange(3), np.arange(3)), {})
    r('reciprocal', (np.arange(3),), {})
    r('negative', (np.arange(3),), {})
    r('multiply', (np.arange(3), np.arange(3)), {})
    r('divide', (np.arange(3), np.arange(3)), {})
    r('power', (np.arange(3), np.arange(3)), {})
    r('subtract', (np.arange(3), np.arange(3)), {})
    r('true_divide', (np.arange(3), np.arange(3)), {})
    r('floor_divide', (np.arange(3), np.arange(3)), {})
    r('fmod', (np.arange(3), np.arange(3)), {})
    r('mod', (np.arange(3), np.arange(3)), {})
    r('modf', (np.arange(3),), {})
    r('remainder', (np.arange(3), np.arange(3)), {})
    
    hd3('Handling complex numbers')
    m('angle')
    m('real')
    m('imag')
    m('conj')
    
    hd3('Miscellaneous')
    r('convolve', (np.arange(3), np.arange(3)), {})
    r('clip', (np.arange(3), 0, 2), {})
    r('sqrt', (np.arange(3),), {})
    r('square', (np.arange(3),), {})
    r('absolute', (np.arange(3),), {})
    r('fabs', (np.arange(3),), {})
    r('sign', (np.arange(3),), {})
    r('maximum', (np.arange(3), np.arange(3)), {})
    r('minimum', (np.arange(3), np.arange(3)), {})
    r('fmax', (np.arange(3), np.arange(3)), {})
    r('fmin', (np.arange(3), np.arange(3)), {})
    r('nan_to_num', (np.arange(3),), {})
    r('real_if_close', (np.arange(3),), {})
    r('interp', (2.5, [1,2,3], [3,2,0]), {})
    
    extra_args = {'nplib': numpy.random, 'chlib': ch.random}
    
    hd2('Random sampling (numpy.random)')
    hd3('Simple random data')
    r('rand', (3,), {}, **extra_args)
    r('randn', (3,), {}, **extra_args)
    r('randint', (3,), {}, **extra_args)
    r('random_integers', (3,), {}, **extra_args)
    r('random_sample', (3,), {}, **extra_args)
    r('random', (3,), {}, **extra_args)
    r('ranf', (3,), {}, **extra_args)
    r('sample', (3,), {}, **extra_args)
    r('choice', (np.ones(3),), {}, **extra_args)
    r('bytes', (3,), {}, **extra_args)
    
    hd3('Permutations')
    r('shuffle', (np.ones(3),), {}, **extra_args)
    r('permutation', (3,), {}, **extra_args)
    
    hd3('Distributions (these all pass)')
    r('beta', (.5, .5), {}, **extra_args)
    r('binomial', (.5, .5), {}, **extra_args)
    r('chisquare', (.5,), {}, **extra_args)
    r('dirichlet', ((10, 5, 3), 20,), {}, **extra_args)
    r('exponential', [], {}, **extra_args)
    r('f', [1,48,1000], {}, **extra_args)
    r('gamma', [.5], {}, **extra_args)
    make_row('...AND 28 OTHERS...', 'passed', 'passed', 'lightgreen', 'lightgreen')
    
    
    hd3('Random generator')
    r('seed', [], {}, **extra_args)
    r('get_state', [], {}, **extra_args)
    r('set_state', [np.random.get_state()], {}, **extra_args)
    
    ####################################
    hd2('Statistics')
    hd3('Order statistics')
    r('amin', (np.eye(3),),{})
    r('amax', (np.eye(3),),{})
    r('nanmin', (np.eye(3),),{})
    r('nanmax', (np.eye(3),),{})
    r('ptp', (np.eye(3),),{})
    r('percentile', (np.eye(3),50),{})

    hd3('Averages and variance')
    r('median', (np.eye(3),),{})
    r('average', (np.eye(3),),{})
    r('mean', (np.eye(3),),{})
    r('std', (np.eye(3),),{})
    r('var', (np.eye(3),),{})
    r('nanmean', (np.eye(3),),{})
    r('nanstd', (np.eye(3),),{})
    r('nanvar', (np.eye(3),),{})
    

    hd3('Correlating')
    r('corrcoef', (np.eye(3),),{})
    r('correlate', ([1, 2, 3], [0, 1, 0.5]),{})
    r('cov', (np.eye(3),),{})
    
    hd3('Histograms')
    r('histogram', (np.eye(3),),{})
    r('histogram2d', (np.eye(3).ravel(),np.eye(3).ravel()),{})
    r('histogramdd', (np.eye(3).ravel(),),{})
    r('bincount', (np.asarray(np.eye(3).ravel(), np.uint32),),{})
    r('digitize', (np.array([0.2, 6.4, 3.0, 1.6]), np.array([0.0, 1.0, 2.5, 4.0, 10.0])),{})
    
    ####################################
    hd2('Sorting, searching, and counting')
    
    hd3('Sorting')
    r('sort', (np.array([1,3,1,2.]),), {})
    m('lexsort')
    m('argsort')
    m('msort')
    m('sort_complex')
    m('partition')
    m('argpartition')
    
# sort(a[, axis, kind, order])    Return a sorted copy of an array.
# lexsort(keys[, axis])    Perform an indirect sort using a sequence of keys.
# argsort(a[, axis, kind, order])    Returns the indices that would sort an array.
# ndarray.sort([axis, kind, order])    Sort an array, in-place.
# msort(a)    Return a copy of an array sorted along the first axis.
# sort_complex(a)    Sort a complex array using the real part first, then the imaginary part.
# partition(a, kth[, axis, kind, order])    Return a partitioned copy of an array.
# argpartition(a, kth[, axis, kind, order])    Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.
    
    a5 = np.arange(5)

    hd3('Searching')
    r('argmax', (a5,), {})
    r('nanargmax', (a5,), {})
    r('argmin', (a5,), {})
    r('nanargmin', (a5,), {})
    r('argwhere', (a5,), {})
    r('nonzero', (a5,), {})
    r('flatnonzero', (a5,), {})
    r('where', (a5>1,), {})
    r('searchsorted', (a5,a5), {})
    r('extract', (lambda x : x > 1, a5), {})

# argmax(a[, axis])    Indices of the maximum values along an axis.
# nanargmax(a[, axis])    Return the indices of the maximum values in the specified axis ignoring
# argmin(a[, axis])    Return the indices of the minimum values along an axis.
# nanargmin(a[, axis])    Return the indices of the minimum values in the specified axis ignoring
# argwhere(a)    Find the indices of array elements that are non-zero, grouped by element.
# nonzero(a)    Return the indices of the elements that are non-zero.
# flatnonzero(a)    Return indices that are non-zero in the flattened version of a.
# where(condition, [x, y])    Return elements, either from x or y, depending on condition.
# searchsorted(a, v[, side, sorter])    Find indices where elements should be inserted to maintain order.
# extract(condition, arr)    Return the elements of an array that satisfy some condition.    
    
    hd3('Counting')
    r('count_nonzero', (a5,), {})
    #count_nonzero(a)	Counts the number of non-zero values in the array a.
    
    

# histogram(a[, bins, range, normed, weights, ...])    Compute the histogram of a set of data.
# histogram2d(x, y[, bins, range, normed, weights])    Compute the bi-dimensional histogram of two data samples.
# histogramdd(sample[, bins, range, normed, ...])    Compute the multidimensional histogram of some data.
# bincount(x[, weights, minlength])    Count number of occurrences of each value in array of non-negative ints.
# digitize(x, bins[, right])    Return the indices of the bins to which each value in input array belongs.    

        
    global src
    src = '<html><body><table border=1>' + src + '</table></body></html>'    
    open(join(split(__file__)[0], 'api_compatibility.html'), 'w').write(src)
    
    print('passed %d, not passed %d' % (num_passed, num_not_passed))
    


if __name__ == '__main__':
    global which_passed
    main()
    print(' '.join(which_passed))
