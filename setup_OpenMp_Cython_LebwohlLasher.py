from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "OpenMP_Cython_LebwohlLasher",
        ["OpenMP_Cython_LebwohlLasher.pyx"],
         extra_compile_args=['-fopenmp'],
         extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()] 
    )
]

setup(
    name="Cython_LebwohlLasher",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)