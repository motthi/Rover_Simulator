from setuptools.extension import Extension
from Cython.Distutils import build_ext
from setuptools import setup
import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('cmotion', sources=["cmotion.pyx"], include_dirs=['.', np.get_include()])],
    include_dirs=[np.get_include()],
    # compiler_directives={'language_level': "3"}
)
