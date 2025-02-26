from setuptools import setup,Extension
from Cython.Build import cythonize
import numpy

# setup(
#     name="OpenMP_Cython_LebwohlLasher",  # 模块名称
#     ext_modules=cythonize("OpenMP_Cython_LebwohlLasher.pyx", compiler_directives={'language_level': "3"},include_dirs=[numpy.get_include()])
# )
ext_modules = [
    Extension(
        "OpenMP_Cython_LebwohlLasher",
        ["OpenMP_Cython_LebwohlLasher.pyx"],
         extra_compile_args=['-fopenmp'],
         extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]  # 确保 NumPy 头文件路径正确
    )
]

# 运行 setup
setup(
    name="Cython_LebwohlLasher",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)