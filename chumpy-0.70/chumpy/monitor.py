'''
Logging service for tracking dr tree changes from root objective 
and record every step that incrementally changes the dr tree

'''
import os, sys, time
import json
import psutil

import scipy.sparse as sp
import numpy as np
from . import reordering

_TWO_20 = float(2 **20)

'''
memory utils

'''
def pdb_mem():
    from .monitor import get_current_memory
    mem = get_current_memory()
    if mem > 7000:
        import pdb;pdb.set_trace()

def get_peak_mem():
    '''
    this returns peak memory use since process starts till the moment its called
    '''
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

def get_current_memory():
    p = psutil.Process(os.getpid())
    mem = p.memory_info()[0]/_TWO_20

    return mem

'''
Helper for Profiler
'''

def build_cache_info(k, v, info_dict):
    if v is not None:
        issparse = sp.issparse(v)
        size = v.size
        if issparse:
            nonzero = len(v.data)
        else:
            nonzero = np.count_nonzero(v)
        info_dict[k.short_name] = {
            'sparse': issparse,
            'size' : str(size),
            'nonzero' : nonzero,
        }


def cache_info(ch_node):
    result = {}
    if isinstance(ch_node, reordering.Concatenate) and hasattr(ch_node, 'dr_cached') and len(ch_node.dr_cached) > 0:
        for k, v in ch_node.dr_cached.items():
            build_cache_info(k, v, result)
    elif len(ch_node._cache['drs']) > 0:
        for k, v in ch_node._cache['drs'].items():
            build_cache_info(k, v, result)

    return result

class DrWrtProfiler(object):
    base_path = os.path.abspath('profiles')
    
    def __init__(self, root, base_path=None):
        self.root = root.obj
        self.history = []

        ts = time.time()
        if base_path:
            self.base_path = base_path
                       
        self.path = os.path.join(self.base_path, 'profile_%s.json' % str(ts))
        self.root_path = os.path.join(self.base_path, 'root_%s.json' % str(ts))

        
        with open(self.root_path, 'w') as f:
            json.dump(self.dump_tree(self.root), f, indent=4)

    def dump_tree(self, node):
        if not hasattr(node, 'dterms'):
            return []

        node_dict = self.serialize_node(node, verbose=False)
        if hasattr(node, 'visited') and node.visited:
            node_dict.update({'indirect':True})
            return node_dict

        node.visited = True
        children_list = []
        for dterm in node.dterms:
            if hasattr(node, dterm):
                child = getattr(node, dterm)
                if hasattr(child, 'dterms') or hasattr(child, 'terms'):
                    children_list.append(self.dump_tree(child))
        node_dict.update({'children':children_list})
        return node_dict

    def serialize_node(self, ch_node, verbose=True):
        node_id = id(ch_node)
        name = ch_node.short_name
        ts = time.time()
        status = ch_node._status
        mem = get_current_memory()
        node_cache_info = cache_info(ch_node)

        rec = {
            'id': str(node_id),
            'indirect' : False,
        }
        if verbose:
            rec.update({
                'name':name,
                'ts' : ts,
                'status':status,
                'mem': mem,
                'cache': node_cache_info,
            })
        return rec

    def show_tree(self, label):
        '''
        show tree from the root node
        '''
        self.root.show_tree_cache(label)

    def record(self, ch_node):
        '''
        Incremental changes
        '''
        rec = self.serialize_node(ch_node)
        self.history.append(rec)

    def harvest(self):
        print('collecting and dump to file %s' % self.path)
        with open(self.path, 'w') as f:
            json.dump(self.history, f, indent=4)