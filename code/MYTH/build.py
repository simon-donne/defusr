
import os
import torch
import torch.utils.ffi
import numpy

force_cuda = True

def main():
    strBasepath = os.path.split(os.path.abspath(__file__))[0] + '/../'
    strHeaders = []
    strSources = []
    strDefines = []
    strObjects = []

    if force_cuda or torch.cuda.is_available():
        strHeaders += ['MYTH/include/MYTH.h']
        strSources += ['MYTH/src/MYTH.c']
        strDefines += [('WITH_CUDA', None)]
        strObjects += ['MYTH/lib/libMYTH_cu.so']

    objectExtension = torch.utils.ffi.create_extension(
        name='MYTH',
        headers=strHeaders,
        sources=strSources,
        verbose=False,
        with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines),
        package=False,
        relative_to=strBasepath,
        include_dirs=['/usr/local/cuda/include', strBasepath + 'MYTH/include/'],
        define_macros=strDefines,
        extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects]
    )

    if __name__ == '__main__':
        objectExtension.build()

if __name__ == "__main__":
    main()
