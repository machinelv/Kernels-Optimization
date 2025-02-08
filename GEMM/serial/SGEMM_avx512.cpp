#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <cstdlib>
#include <mkl.h>
#include <vector>
#include <memory>  // 包含智能指针的头文件

using namespace std;

#define DEBUG(x) cout << #x << "= " << x << endl;


/*****
MKL library:
void cblas_sgemm(
    const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, 
    const BLASINT M, const BLASINT N, const BLASINT K, 
    const double alpha, 
    const double *A, const BLASINT lda, 
    const double *B, const BLASINT ldb, 
    const double beta, 
    double *C, const BLASINT ldc);
****/

template <typename T>
void mkl_dgemm(
    const T *A, const T *B, T *out, 
    const int M, const int K, const int N) 
{   
    int lda = K, ldb = N, ldc = N; 
    T alpha = 1.0, beta = 0;
    if constexpr (std::is_same<T, float>::value) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 
                    alpha, A, lda, B, ldb, 
                    beta, out, ldc);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 
                    alpha, A, lda, B, ldb, 
                    beta, out, ldc);
    }
}



//Cache Partition
#define MC 128
#define KC 128
#define NC 128

//Register Partition
#define MR 64
#define NR 64

template <typename T>
void gemm_AVX512_v1(
    const T *A, const T *B, T *out, 
    const int M, const int K, const int N) 
{  
  int lda = K, ldb = N, ldc = N; 
  T alpha = 1.0, beta = 0;
  //Memory to Cache
  


}



template <typename T>
void gemm_test(const T *A, const T *B, T *C, const int M, const int K, const int N, const int algorithm, const int ntest) {
  int alpha = 1, beta = 0;

  double elapse_time_all = 0;


  for (int i = 0; i < ntest; i++) {
    for (int j = 0; j < M * N; j++) {
      C[j] = 0.0f;
    }
    double start_time = timestamp();
    if (algorithm == 0) {
      kml_gemm(A, B, C, M, K, N);
    } else {
      gemm_AVX512_v1(A, B, C, M, K, N, 1);
      // svfloat64_t ss;
      // svmla_f64_m(svptrue_b64(), ss, ss, ss);
    }
    double end_time = timestamp();
    elapse_time_all += end_time - start_time;
  }
  


  double elapse_time = elapse_time_all / ntest;

  double nflops = (double) M * N * K * 2.0f;
  double gflops = nflops * 1.0e-9 / elapse_time;
  // SVE: 256B*2
  // Clock: 2900.0000 MHz
  //  

  if (algorithm == 0) {
      printf("SGEMM: KML ; ");
  }
  else {
      printf("Block Size = %d x %d x %d ; Kernel Size = %d x %d x %d \n", MC, KC, NC, MR, KC, NR);
      printf("SGEMM: SME ; ");
  }
  // printf("Matrix Size: (%d,%d)x(%d,%d): Theoretical Performance %7.2lf GFlops \n", M,K,K,N, gflops_theory);

  printf("Matrix Size: (%dx%d)X(%dx%d) ; Elapse time (ms): %lf ; GFlops: %7.2lf \n", M,K,K,N, elapse_time * 1000, gflops);

}


template <typename T>
void ans_verify(const T *ans, const T *C, const int M, const int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double err = (ans[i * N + j] - C[i * N + j]) / ans[i * N + j];
      if (err > 0.03 || err < -0.03) {
        printf("The result is incorrect!!\n");
        return;
      }
    }
  }
  printf("The varification passed.\n");
}


int main(int argc, char *argv[]) {
  int M, K, N;
  M = K = N = MM_SIZE;

  double *A, *B, *C, *ans;
  A = (double *)aligned_alloc(64, sizeof(double) * M * K);
  assert(A != NULL);
  B = (double *)aligned_alloc(64, sizeof(double) * K * N);
  assert(B != NULL);
  C = (double *)aligned_alloc(64, sizeof(double) * M * N);
  assert(C != NULL);
  ans = (double *)aligned_alloc(64, sizeof(double) * M * N);
  assert(ans != NULL);

  for (long i = 0; i < (size_t) M * K; i++) A[i] = (double)(i % 10 + 1) + ((double)(i % 10 + 1) / 123);

  for (long i = 0; i < K * N; i++) B[i] = (double)(i / (K * N) + 1) + ((double)(i / (K * N) + 1)  / 123 );

  // 担心上面的空间还没创建完成，就启动了计时模块
  usleep(30);


  gemm_test(A, B, ans, M, K, N, 0, LOOP_NUM);
  gemm_test(A, B, C, M, K, N, 1, LOOP_NUM);
  
  ans_verify(ans, C, M, N);

#ifdef _DEBUG
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        printf("%lf ", ans[i * N + j]);
    }
    printf("\n");
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        printf("%lf ", C[i * N + j]);
    }
    printf("\n");
  }
#endif

  free(A);
  free(B);
  free(C);
  free(ans);

  return 0;
}