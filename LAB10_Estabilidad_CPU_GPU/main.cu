#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#include "tools.h"

#define CHECK

// Kernel
__global__ void mikernel(float a, float *x, float *y, float *s, long n){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        s[i] = a * x[i] + y[i];
    }
}

void cpu(float a, float *x, float *y, float *z, long n);
void init_vec(float *a, long n, float c);
void print_vec(float *a, long n, const char *msg);
void print_gpu_specs(int dev);
int check_result(float *comp, float *ref, long n);

int main(int argc,char **argv){
	if(argc != 7){
        fprintf(stderr, "error ejecutar como ./prog <dev> <n> <mode> <nt> <GPU_BS> <GPU_NB>\n"
                "\tdev      : GPU ID (0,1,...etc) --> check with `nvidia-smi`\n"
                "\tmode     : 0 -> CPU   1 -> GPU\n"
                "\tnt       : number of CPU threads\n"
                "\tGPU_BS   : CUDA Block Size (min=1, max=1024) for mode=1\n"
                "\tGPU_NB   : Number of CUDA blocks for mode=1, GPU_NB=-1 -> adaptive\n");
		exit(EXIT_FAILURE);
	}
    int dev, m, nt, bs, numBlocks;
    long n;
    // punteros HOST
    float a = 1.0f, *x,  *y,  *s1, *s2;
    // punteros DEVICE
    float   *dx, *dy, *ds;
    // obtener argumentos
    dev = atoi(argv[1]);
    n = atoi(argv[2]);
    m = atoi(argv[3]);
    nt = atoi(argv[4]);
    bs = atoi(argv[5]);
    numBlocks = atoi(argv[6]);
    printf("CONFIG:  dev=%i  n=%i   mode=%i   nt=%i   bs=%i   numBlocks=%i\n", dev, n, m, nt, bs, numBlocks); fflush(stdout);
    printf("n=%lu elements -> %.2f GBytes\n", n, n*sizeof(float)/1e9);

    // omp thread setting
    omp_set_num_threads(nt);

    // inicializar arreglos en Host (CPU)
    x = (float*)malloc(sizeof(float)*n);
    y = (float*)malloc(sizeof(float)*n);
    s1 = (float*)malloc(sizeof(float)*n);
    s2 = (float*)malloc(sizeof(float)*n);

    init_vec(x, n, 1);
    print_vec(x, n, "vector x");
    init_vec(y, n, 2);
    print_vec(y, n, "vector y");
    init_vec(s1, n, 0);

    // allocar memoria en device  (GPU)
    // cudaMalloc( puntero del puntero, bytes)
    CHECK_CUDA_ERROR(cudaMalloc(&dx, sizeof(float) * n));
    CHECK_CUDA_ERROR(cudaMalloc(&dy, sizeof(float) * n));
    CHECK_CUDA_ERROR(cudaMalloc(&ds, sizeof(float) * n));

    // copiar de Host -> Device
    //cudaMemcpy(destino, origen, bytes, direccion)
    CHECK_CUDA_ERROR(cudaMemcpy(dx, x, sizeof(float)*n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dy, y, sizeof(float)*n, cudaMemcpyHostToDevice));
    //cudaMemcpy(ds, s, sizeof(float)*n, cudaMemcpyHostToDevice)

    dim3 block(bs, 1, 1);
    dim3 grid((n + bs -1)/bs, 1, 1);
    if(numBlocks > 0){
        grid = dim3(numBlocks, 1, 1);
    }

    double t1, t2;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    printf("\n");
	if(m){
        // obtener info de GPU
        print_gpu_specs(dev);
		printf("[GPU] grid=(%5i, %i, %i) block(%5i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z); fflush(stdout);
		printf("GPU SAXPY........................."); fflush(stdout);
	    cudaEventRecord(start);
        t1 = omp_get_wtime();
		mikernel<<<grid, block>>>(a, dx, dy, ds, n);
	    cudaDeviceSynchronize();
        t2 = omp_get_wtime();
        CHECK_LAST_CUDA_ERROR();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
	}
	else{
		printf("[CPU] nt=%2i\n", nt); fflush(stdout);
		printf("CPU SAXPY........................."); fflush(stdout);
	    cudaEventRecord(start);
        t1 = omp_get_wtime();
		cpu(a, x, y, s2, n);	
        t2 = omp_get_wtime();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
	}
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("ok: %f (%f) secs\n\n", milliseconds/1000.0f, t2-t1); fflush(stdout);
	if(m){
        printf("[GPU] Result to Host.............."); fflush(stdout);
		cudaMemcpy(s1, ds, sizeof(float)*n, cudaMemcpyDeviceToHost);
        printf("done\n"); fflush(stdout);
	}
	print_vec(s1, n, "vector S");		
    printf("\n");

    // CHECK RESULT
    #ifdef CHECK
    if(m){
        // check GPU result against CPU Reference
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("GOLD CPU (nt=%2i)..................", nt); fflush(stdout);
        t1 = omp_get_wtime();
        cpu(a, x, y, s2, n);	
        t2 = omp_get_wtime();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gold_ms = 0;
        cudaEventElapsedTime(&gold_ms, start, stop);
        printf("ok: %f (%f) secs\n", gold_ms/1000.0f, t2-t1); fflush(stdout);
        print_vec(s1, n, "Gold S");		
        check_result(s1, s2, n);
    }
    #endif
}



void cpu(float a, float *x, float *y, float *s, long n){
    #pragma omp parallel for
	for(int i=0;i<n;i++){
		s[i]=a*x[i]+y[i];
	}
}
void init_vec(float *a, long n, float c){
	#pragma omp parallel for
	for(int i=0; i<n; ++i){
		a[i] = c*i;
	}
}

void print_vec(float *a, long n, const char *msg){
    if(n > 32){ return; }
    printf("%s\n[", msg);
    for(int i=0; i<n; ++i){
        printf("%f ", a[i]);
    }
    printf("]\n");
}

#define EPSILON 1e-5
int check_result(float *comp, float *ref, long n){
    printf("Checking result......."); fflush(stdout);
    for(long i=0; i<n; ++i){
        if(fabs(comp[i] - ref[i]) > EPSILON){
            printf("comp[%i]=%f   !=   ref[%i]=%f\n", i, comp[i], i, ref[i]);
            printf("failed\n"); fflush(stdout);
            return 0;
        }
    }
    printf("pass\n"); fflush(stdout);
    return 1;
}

void print_gpu_specs(int dev){
    cudaDeviceProp prop;
    printf("Requesting GPU properties......."); fflush(stdout);
    cudaGetDeviceProperties(&prop, dev);
    printf("done\n"); fflush(stdout);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Memory:                       %f GB\n", prop.totalGlobalMem/(1024.0*1024.0*1024.0));
    printf("  Multiprocessor Count:         %d SMs\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %s\n", prop.concurrentKernels == 1? "yes" : "no");
    printf("  Memory Clock Rate:            %d MHz\n", prop.memoryClockRate);
    printf("  Memory Bus Width:             %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth:        %f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}
