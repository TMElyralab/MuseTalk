__author__ = 'matt'

from . import ch
import numpy as np
from .utils import row, col
import scipy.sparse as sp
import scipy.special

class Interp3D(ch.Ch):
    dterms = 'locations'
    terms = 'image'

    def on_changed(self, which):
        if 'image' in which:
            self.gx, self.gy, self.gz = np.gradient(self.image)

    def compute_r(self):
        locations = self.locations.r.copy()
        for i in range(3):
            locations[:,i] = np.clip(locations[:,i], 0, self.image.shape[i]-1)
        locs = np.floor(locations).astype(np.uint32)
        result = self.image[locs[:,0], locs[:,1], locs[:,2]]
        offset = (locations - locs)
        dr = self.dr_wrt(self.locations).dot(offset.ravel())
        return result + dr

    def compute_dr_wrt(self, wrt):
        if wrt is self.locations:
            locations = self.locations.r.copy()
            for i in range(3):
                locations[:,i] = np.clip(locations[:,i], 0, self.image.shape[i]-1)
            locations = locations.astype(np.uint32)

            xc = col(self.gx[locations[:,0], locations[:,1], locations[:,2]])
            yc = col(self.gy[locations[:,0], locations[:,1], locations[:,2]])
            zc = col(self.gz[locations[:,0], locations[:,1], locations[:,2]])

            data = np.vstack([xc.ravel(), yc.ravel(), zc.ravel()]).T.copy()
            JS = np.arange(locations.size)
            IS = JS // 3

            return sp.csc_matrix((data.ravel(), (IS, JS)))


class gamma(ch.Ch):
    dterms = 'x',

    def compute_r(self):
        return scipy.special.gamma(self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:
            d = scipy.special.polygamma(0, self.x.r)*self.r
            return sp.diags([d.ravel()], [0])

# This function is based directly on the "moment" function
# in scipy, specifically in mstats_basic.py.
def moment(a, moment=1, axis=0):
    if moment == 1:
        # By definition the first moment about the mean is 0.
        shape = list(a.shape)
        del shape[axis]
        if shape:
            # return an actual array of the appropriate shape
            return ch.zeros(shape, dtype=float)
        else:
            # the input was 1D, so return a scalar instead of a rank-0 array
            return np.float64(0.0)
    else:
        mn = ch.expand_dims(a.mean(axis=axis), axis)
        s = ch.power((a-mn), moment)
        return s.mean(axis=axis)
