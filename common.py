from PIL import Image
import pickle
from io import BytesIO

import numpy as np


def pkwrite(obj, fn):
    """Write object `obj` to file name `fn` in pickle format."""
    with open(fn, 'wb') as fd:
        pickle.dump(obj, fd)


def pkread(fn):
    """Read pickled file `fn`."""
    with open(fn, 'rb') as fd:
        obj = pickle.load(fd)
    return obj


def file2imgarr(fn, shape=None, convert=None):
    """Read image file `fn`, convert to numpy array.

    Parameters
    ----------
    fn : str
        file name
    shape : {None, 2-tuple}
        reshape to shape=(height, width) = (nrows, ncols)
    convert : passed to ``PIL.Image.convert(mode=convert)``

    Returns
    -------
    imgarr : np.array
        usually shape (height, width, 3) for RBG and (height, width) for
        gray (convert='L')
    """
    im = Image.open(fn)
    if convert:
        im = im.convert(convert)
    if shape is None:
        return np.array(im)
    else:
        return np.array(im.resize((shape[1], shape[0]), 3))


# > 10x size compression with quality=75 (PIL default), we can go as low as 30%
# w/o significant visual loss of quality
def imgarr2jpegstr(imgarr, quality=50):
    """JPEG-compress image.

    Parameters
    ----------
    imgarr : np.array
        see :func:`file2imgarr`
    quality : int
        compression quality, 1 (worst) ... 95 (best)

    Returns
    -------
    jpegstr : byte string
    """
    fd = BytesIO()
    im = Image.fromarray(imgarr)
    im.save(fd, format='jpeg', optimize=True, quality=quality)
    fd.seek(0)
    jpegstr = fd.read()
    fd.close()
    return jpegstr


def jpegstr2imgarr(jpegstr, **kwds):
    """Inverse of :func:`imgarr2jpegstr`.

    We use an in-memory file object, so you can use **kwds which we pass to
    :func:`file2imgarr`
    """
    return file2imgarr(BytesIO(jpegstr), **kwds)


def file2jpegstr(fn, shape=None, convert=None, quality=None):
    """Read image file, convert to jpegstr. """
    return imgarr2jpegstr(file2imgarr(fn, shape=shape, convert=convert),
                          quality=quality)
