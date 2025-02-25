from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='stbif_cuda',
    ext_modules=[
        CUDAExtension(
            name='stbif_cuda',
            sources=['./cuda/STBIF_forward.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
