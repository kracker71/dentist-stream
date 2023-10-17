from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='ctc_beamsearch',
    ext_modules=cythonize("ctc_beamsearch.pyx", 
                compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)