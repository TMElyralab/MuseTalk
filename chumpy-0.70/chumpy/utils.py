"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""
import scipy.sparse as sp 
import numpy as np

def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))

class timer(object):
    def time(self):
        import time
        return time.time()
    def __init__(self):
        self._elapsed = 0
        self._start = self.time()
    def __call__(self):
        if self._start is not None:
            return self._elapsed + self.time() - self._start
        else:
            return self._elapsed
    def pause(self):
        assert self._start is not None
        self._elapsed += self.time() - self._start
        self._start = None
    def resume(self):
        assert self._start is None
        self._start = self.time()

def dfs_do_func_on_graph(node, func, *args, **kwargs):
    '''
    invoke func on each node of the dr graph
    '''
    for _node in node.tree_iterator():
        func(_node, *args, **kwargs)


def sparse_is_desireable(lhs, rhs):
    '''
    Examines a pair of matrices and determines if the result of their multiplication should be sparse or not.
    '''
    return False
    if len(lhs.shape) == 1:
        return False
    else:
        lhs_rows, lhs_cols = lhs.shape

    if len(rhs.shape) == 1:
        rhs_rows = 1
        rhs_cols = rhs.size
    else:
        rhs_rows, rhs_cols = rhs.shape

    result_size = lhs_rows * rhs_cols

    if sp.issparse(lhs) and sp.issparse(rhs):
        return True
    elif sp.issparse(lhs):
        lhs_zero_rows = lhs_rows - np.unique(lhs.nonzero()[0]).size
        rhs_zero_cols = np.all(rhs==0, axis=0).sum()
        
    elif sp.issparse(rhs):
        lhs_zero_rows = np.all(lhs==0, axis=1).sum()
        rhs_zero_cols = rhs_cols- np.unique(rhs.nonzero()[1]).size
    else:
        lhs_zero_rows = np.all(lhs==0, axis=1).sum()
        rhs_zero_cols = np.all(rhs==0, axis=0).sum()

    num_zeros = lhs_zero_rows * rhs_cols + rhs_zero_cols * lhs_rows - lhs_zero_rows * rhs_zero_cols

    # A sparse matrix uses roughly 16 bytes per nonzero element (8 + 2 4-byte inds), while a dense matrix uses 8 bytes per element. So the break even point for sparsity is 50% nonzero. But in practice, it seems to be that the compression in a csc or csr matrix gets us break even at ~65% nonzero, which lets us say 50% is a conservative, worst cases cutoff.
    return (float(num_zeros) / float(size)) >= 0.5


def convert_inputs_to_sparse_if_necessary(lhs, rhs):
    '''
    This function checks to see if a sparse output is desireable given the inputs and if so, casts the inputs to sparse in order to make it so.
    '''
    if not sp.issparse(lhs) or not sp.issparse(rhs):
        if sparse_is_desireable(lhs, rhs):
            if not sp.issparse(lhs):
                lhs = sp.csc_matrix(lhs)
                #print "converting lhs into sparse matrix"
            if not sp.issparse(rhs):
                rhs = sp.csc_matrix(rhs)
                #print "converting rhs into sparse matrix"
    return lhs, rhs
