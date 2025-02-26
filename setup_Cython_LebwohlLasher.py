from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "Cython_LebwohlLasher",
        ["Cython_LebwohlLasher.pyx"],
        include_dirs=[numpy.get_include()]  
    )
]

# 运行 setup
setup(
    name="Cython_LebwohlLasher",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)