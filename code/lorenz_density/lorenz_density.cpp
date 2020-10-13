// density/bakers_density.cpp
#include<cstdio>
#include<cassert>
#include<algorithm>

#include"histogram.h"
#include"lorenz.h"

struct ReturnXZ {
    __device__ __forceinline__
    static ValueWeight eval(double xyz[3]) {
        return ValueWeight{xyz[0], xyz[2], 1.0};
    }
};

int main() {
    uint32_t iDevice;
    assert(fread(&iDevice, sizeof(uint32_t), 1, stdin) == 1);
    cudaSetDevice(iDevice);
    fprintf(stderr, "Set To Device %u\n", iDevice);

    uint32_t randSeed;
    assert(fread(&randSeed, sizeof(uint32_t), 1, stdin) == 1);
    srand(randSeed);

    const int nx = 2048, nz = 2048;
    Counter<3> counter(nx, nz, -20., 0., 40./nx, 50./nz);

    float parameters[5];
    assert(fread(parameters, sizeof(float), 5, stdin) == 5);
    Lorenz lorenz{parameters[0], parameters[1], parameters[2], parameters[3]};
    ReturnXZ obj;
    fprintf(stderr, "Ready with parameters (%f %f %f) with dt %f for %f steps\n",
            parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]);

    uint32_t nIters;
    assert(fread(&nIters, sizeof(uint32_t), 1, stdin) == 1);

    for (uint32_t iIter = 0; iIter < nIters; iIter++) {
        if (iIter % 128 == 0) {
            fprintf(stderr, "%u/%u iterations\n", iIter, nIters);
        }

        counter.init(lorenz, 80000);
        counter.run(lorenz, obj, static_cast<int>(parameters[4]), false);

        counter.run(lorenz, obj, 1, true);
    }

    fwrite(counter.counts, sizeof(double), nx * nx, stdout);
    return 0;
}
