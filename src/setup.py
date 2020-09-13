from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[
        # Extension("cnumpy_utils", ["planning/utils/cnumpy_utils.pyx"]),
        # Extension("math_utils", ["planning/utils/math_utils.pyx"]),
        Extension("poly1d", ["planning/utils/optimal_control/poly1d.pyx"])
      ]
)