from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='softgroup',
        version='1.1',
        description='SoftGroup: SoftGroup for 3D Instance Segmentation [CVPR 2022]',
        author='Thang Vu',
        author_email='thangvubk@kaist.ac.kr',
        packages=find_packages(),
        package_dir={'':'src'},
        package_data={'softgroup.ops': ['*/*.so']},
        ext_modules=[
            CUDAExtension(
                name='softgroup.ops.ops',
                sources=[
                    'src/softgroup/ops/src/softgroup_api.cpp', 'src/softgroup/ops/src/softgroup_ops.cpp',
                    'src/softgroup/ops/src/cuda.cu'
                ],
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                })
        ],
        cmdclass={'build_ext': BuildExtension})
