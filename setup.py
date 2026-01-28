from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                "kernels/laplacian.cu",
                "kernels/module.cpp"
            ],
            libraries=['nvToolsExt'],
            extra_compile_args={"cxx": ["-O3"],
                                "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
