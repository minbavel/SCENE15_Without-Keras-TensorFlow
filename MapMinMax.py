
import numpy as np

class settings:
    def __init__(self, xmax, xmin, ymax, ymin, yrange, xrange):
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.yrange = yrange
        self.xrange = xrange
        pass


def mapminmax(x, ymin=-1.0, ymax=1.0):
    return create(x, ymin, ymax)


def create(x, ymin, ymax):
    xrows = x.shape[0]
    xmin = x.min(1)
    xmax = x.max(1)

    xrange = xmax - xmin
    yrows = xrows
    yrange = ymax - ymin

    gain = yrange / xrange

    fix = np.nonzero(~np.isfinite(xrange) | (xrange == 0))

    if(not all(fix)):
        None
    else:
        gain[fix] = 1
        xmin[fix] = ymin

    return [mapminmax_apply(x, xrange, xmin, yrange, ymin),
            settings(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin, yrange=yrange, xrange=xrange)]


def mapminmax_apply(x, xrange, xmin, yrange, ymin):
    gain = yrange / xrange

    fix = np.nonzero(~np.isfinite(xrange) | (xrange == 0))
    if(not all(fix)):
        None
    else:
        gain[fix] = 1
        xmin[fix] = ymin

    cd = np.multiply((np.ones((x.shape[0], x.shape[1]))), xmin.values.reshape(x.shape[0], 1))
    a = x - cd

    b = np.multiply((np.ones((x.shape[0], x.shape[1]))), gain.values.reshape(x.shape[0], 1))
    return np.multiply(a, b) + ymin


class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return x * self.slope + self.intercept
    def reverse(self, y):
        return (y-self.intercept) / self.slope
    
def mapminmax_rev(x, ymin=-1, ymax=+1):
    x = np.asanyarray(x)
    xmax = x.max(axis=-1)
    xmin = x.min(axis=-1)
    if (xmax==xmin).any():
        raise ValueError("some rows have no variation")
    slope = ((ymax-ymin) / (xmax - xmin))[:,np.newaxis]
    intercept = (-xmin*(ymax-ymin)/(xmax-xmin))[:,np.newaxis] + ymin
    ps = MapMinMaxApplier(slope, intercept)
    return ps(x), ps