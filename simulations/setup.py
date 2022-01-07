from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['smcTranslocator_MovingBarrier.pyx',
                           'smcTranslocator_SimpleGene.pyx']
                         )
)

# Run $ python setup.py build_ext --inplace
#     $ python setup.py install
