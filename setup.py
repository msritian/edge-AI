from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='bitwise_ops',
    ext_modules=[
        CppExtension('bitwise_ops', ['bitwise_kernel.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
