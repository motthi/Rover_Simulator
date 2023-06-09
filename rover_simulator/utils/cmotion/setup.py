from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

setup(
    name="cmotion",
    ext_modules=cythonize([
        Extension("cmotion", sources=["cmotion.pyx"], include_dirs=['.', get_include()])
    ])
)
