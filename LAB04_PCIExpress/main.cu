#include <cuda.h>
#include <stdio.h>
#include <omp.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

char* newArrayCPU(long bytes){
    char *p;
    gpuErrchk(cudaMallocHost((void**)&p, bytes));
    return p;
}

char* newArrayGPU(long bytes){
    char *p;
    gpuErrchk(cudaMalloc(&p, bytes));
    return p;
}

void copyFromCPUToGPU(char *host, char *device, long bytes){
    gpuErrchk(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
}

void copyFromGPUToCPU(char *host, char *device, long bytes){
    gpuErrchk(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}

// NOTA, puede medir tiempo de esta manera.
// double t1 = omp_get_wtime();
// ... instrucciones ...
// double t2 = omp_get_wtime();
// double ttotal = t2-t1;
int main(int argc, char **argv){
    if(argc != 3){
        fprintf(stderr, "ejecutar como ./prog n r\nn = tamaño (GBytes)\nr = repeticiones\n");
        exit(EXIT_FAILURE);
    }
    double GBytes = atof(argv[1]);
    long n = (long)(GBytes*1e9);
    printf("n = %f GBytes (%lu bytes)\n", GBytes, n);

    printf("creating CPU array.......");  fflush(stdout);
    char* host = newArrayCPU(n);
    printf("done\n");  fflush(stdout);
    printf("creating CPU array.......");  fflush(stdout);
    char* dev = newArrayGPU(n);
    printf("done\n");  fflush(stdout);

    r = atoi(argv[2]);
    for(int i = 0; i < r; i++){
        double t1 = omp_get_wtime();
        copyFromCPUToGPU(host,dev,n);
        copyFromGPUToCPU(host,dev,n);
        double t2 = omp_get_wtime();
        double total = t2-t1;
        printf("El tiempo total es: ", total);
    }

    // PROGRAMAR LAB PCIe Aqui
    // BEGIN
    // .....
    // .....
    // END
    return EXIT_SUCCESS;
}
