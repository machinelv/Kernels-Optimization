#define _XOPEN_SOURCE   600
#define _POSIX_C_SOURCE 200112L
#include <time.h>
#include <sys/time.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

#include "arm_sve.h"
#include "kblas.h"
#include "omp.h"


#define VEC_LEN 4
#define CEIL4(i) (((i) + 3) / 4 * 4)
#define CEIL8(i) (((i) + 7) / 8 * 8)
#define mul_s(vec, scalar) svmul_n_f64_m(svptrue_b64(), vec, scalar)
#define load_sve(addr) svld1_f64(svptrue_b64(), addr)
#define store_sve(addr, vec) svst1_f64(svptrue_b64(), addr, vec)
#define fma_sve(c, a, b) svmla_f64_m(svptrue_b64(), c, a, b)    // c + (a * b)
#define fmul_sve(a, b) svmul_f64_m(svptrue_b64(), a, b)    //  (a * b)
// #define fms_sve(a, b) svmmla_f64(svptrue_b64(), a, b)    //  sve_sum(sve_mul(a * b))

#define fma_s(c, a, b_scalar) svmla_n_f64_m(svptrue_b64(), c, a, b_scalar)    // c + (a * b)
#define sum_sve(a) svaddv_f64(svptrue_b64(), a)  // sum(a[0], a[1] ...)

#define fmma_s(a,b) sum_sve(fmul_sve(a,b))

//svld1_gather_[s64]offset[_f64](svbool_t pg,  const float64_t *base, svuint64_t offsets)
//svfloat64_t svld1_gather[_u64base]_offset_f64(svbool_t pg, svuint64_t bases, int64_t offset)
#define load_gather(addr, offsets) svld1_gather_u64offset_f64(svptrue_b64(), addr, offsets) 

#define MIN(a, b) ((a) < (b) ? (a) : (b)):q

#define LOOP_NUM 1
#define MM_SIZE 2048

#define Block_M 16
#define Block_K 4*VEC_LEN
#define Block_N 16

double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1.e-6;
}

// Matrix Multiplication: Out = A x B (A:M*K, B:K*N, out: M*N)
// All arrays should have their memory prepared correctly outside this function

/*****
KML library:
void cblas_sgemm(
    const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, 
    const BLASINT M, const BLASINT N, const BLASINT K, 
    const double alpha, 
    const double *A, const BLASINT lda, 
    const double *B, const BLASINT ldb, 
    const double beta, 
    double *C, const BLASINT ldc);
****/

inline static void kml_dgemm(
    const double *A, const double *B, double *out, 
    const int M, const int K, const int N) 
{   
    int lda = K, ldb = N, ldc = N; 
    double alpha = 1.0, beta = 0; 
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 
                alpha, A, lda, B, ldb, 
                beta, out, ldc);
}

#define DEBUG(x) printf("%s=%d\n", #x, (double)x);


static void sve_dgemm(const int M, const int N, const int K, 
    const double alpha, const double *A, const double *B,
    const double beta, double *out)
{  
  for (int i = 0; i < M; i += Block_M) 
    for (int j = 0; j < N; j += Block_N){
      for (int l = 0; l < K; l += Block_K) {

        for (int m = i; m < i + Block_M && m < M; m ++) {
          for (int n = j; n < j + Block_N && n < N; n ++) {
            svfloat64_t out_tmp = svdup_f64(0);
            double b_tmp[VEC_LEN];
            for (int k = l; k < l + Block_K && k < K; k += VEC_LEN) {
              for (int kk = 0; kk < VEC_LEN; kk++) {
                b_tmp[kk] = B[(kk + k) * N + n];            
              }
              out_tmp = fma_sve(out_tmp, load_sve(&A[m * K + k]), load_sve(&b_tmp[0]));
            }
            out[m * N + n] += sum_sve(out_tmp);
          }
        }

      }
    }
}

static void sve_dgemm_v2(uint64_t M, uint64_t N, uint64_t K, 
      const double alpha, const double *inLeft, const double *inRight, 
      const double beta, double *out) {
    uint64_t x, y, z;
    svbool_t p64_all = svptrue_b64();
    uint64_t vl = svcntd();
    uint64_t offsetIN_1, offsetIN_2, offsetIN_3;
    uint64_t offsetOUT_1, offsetOUT_2, offsetOUT_3;
    float64_t *ptrIN_left;
    float64_t *ptrIN_right;
    float64_t *ptrOUT;
    svfloat64_t acc0, acc1, acc2, acc3;
    svfloat64_t inR_0, inR_1;
    svfloat64_t inL_0, inL_1, inL_2, inL_3;

    offsetIN_1 = K;
    offsetIN_2 = 2*K;
    offsetIN_3 = 3*K;
    offsetOUT_1 = N;
    offsetOUT_2 = 2*N;
    offsetOUT_3 = 3*N;

    for (x=0; x<M; x+=4) {
        ptrOUT = &out[x*N];
        for (y=0; y<N; y+=vl) {
            acc0 = svdup_f64(0.0);
            acc1 = svdup_f64(0.0);
            acc2 = svdup_f64(0.0);
            acc3 = svdup_f64(0.0);
            ptrIN_left = &inLeft[x*K];
            ptrIN_right = &inRight[y];
            for (z=0; z<K; z+=2) {
                inR_0 = svld1(p64_all, ptrIN_right);
                inR_1 = svld1(p64_all, &ptrIN_right[offsetOUT_1]);
                inL_0 = svld1rq(p64_all, ptrIN_left);
                inL_1 = svld1rq(p64_all, &ptrIN_left[offsetIN_1]);
                inL_2 = svld1rq(p64_all, &ptrIN_left[offsetIN_2]);
                inL_3 = svld1rq(p64_all, &ptrIN_left[offsetIN_3]);
                acc0 = svmla_lane(acc0, inR_0, inL_0, 0);
                acc0 = svmla_lane(acc0, inR_1, inL_0, 1);
                acc1 = svmla_lane(acc1, inR_0, inL_1, 0);
                acc1 = svmla_lane(acc1, inR_1, inL_1, 1);
                acc2 = svmla_lane(acc2, inR_0, inL_2, 0);
                acc2 = svmla_lane(acc2, inR_1, inL_2, 1);
                acc3 = svmla_lane(acc3, inR_0, inL_3, 0);
                acc3 = svmla_lane(acc3, inR_1, inL_3, 1);
                ptrIN_right += 2*N;
                ptrIN_left += 2;
            }
            svst1(p64_all, ptrOUT, acc0);
            svst1(p64_all, &ptrOUT[offsetOUT_1], acc1);
            svst1(p64_all, &ptrOUT[offsetOUT_2], acc2);
            svst1(p64_all, &ptrOUT[offsetOUT_3], acc3);
            ptrOUT += vl;
        }
    }
}


//Cache Partition
#define MC 128
#define KC 128
#define NC 128

//Register Partition
#define MR 64
#define NR 64


void sve_dgemm_v3(uint64_t M, uint64_t N, uint64_t K, 
      const double alpha, const double *A, const double *B, 
      const double beta, double *C)
{
  //X[c]: Cache buffer
  //X[r]: Register buffer
  const uint64_t vl = svcntd();
#ifdef USE_OMP
  #pragma omp parallel for collapse(5) shared(A, B, C) schedule(dynamic)
#endif
  //Memory to Cache
  for (int mc = 0; mc < M; mc += MC) {
    for (int kc = 0; kc < K; kc += KC) {
       for (int nc = 0; nc < N; nc += NC) {
        //Cache to Register
        for (int nr = 0; nr < NC; nr += NR) {  
          for (int mr = 0; mr < MC; mr += MR){

            //Step 1: Load&Transfer row-major matrix A to column-major Ac
            double A_c[MR*KC];
            double B_c[KC*NR];
            double C_c[MR*NR];

            for (int m = 0; m < MR && m + mr + mc < M; m ++) {
              for (int k = 0; k < KC && k + kc < K; k ++) {
                A_c[k * MR + m] = A[(m + mr + mc) * K + k + kc];
              }
            }

            for (int k = 0; k < KC && k + kc < K; k ++){
              for (int n = 0; n < NR && n + nr + nc < M; n ++) {
                B_c[k * NR + n] = B[(k + kc) * N + n + nr + nc];
              }
            }

            for (int m = 0; m < MR && m + mr + mc < M; m ++){
              for (int n = 0; n < NR && n + nr + nc < M; n ++) {
                C_c[m * NR + n] = C[(m + mr + mc) * N + n + nr + nc];
              }
            }

            for (int kr = 0; kr < KC; kr ++) {
              //MR must be equal to NR
              for (int m = 0; m < MR; m += 4) {
                for (int n = 0; n < NR; n += vl) {

                  int index_m = m + mr + mc;
                  int index_n = n + nr + nc;
                  int index_k = kr + kc;

                  svbool_t pn = svptrue_b64();
                  svfloat64_t va1 = svld1rq(pn, &A_c[kr * MR + m]);
                  svfloat64_t va2 = svld1rq(pn, &A_c[kr * MR + m + 2]);
                  // svfloat64_t va3 = svld1rq(pn, &A_c[kr * MR + m + 4]);
                  // svfloat64_t va4 = svld1rq(pn, &A_c[kr * MR + m + 6]);

                  svfloat64_t vb = svld1_f64(pn, &B_c[kr * NR + n]);
                  // svfloat64_t vb = svld1_f64(pn, &B[index_k * N + index_n]);

                  //Step 2: Load Data into register
                  //Load the previous result in Cr
                  // svfloat64_t vc1 = svld1_f64(pn, &C[(index_m) * N + index_n]);
                  // svfloat64_t vc2 = svld1_f64(pn, &C[(index_m + 1) * N + index_n]);
                  // svfloat64_t vc3 = svld1_f64(pn, &C[(index_m + 2) * N + index_n]);
                  // svfloat64_t vc4 = svld1_f64(pn, &C[(index_m + 3) * N + index_n]);

                  svfloat64_t vc1 = svld1_f64(pn, &C_c[(m) * NR + n]);
                  svfloat64_t vc2 = svld1_f64(pn, &C_c[(m + 1) * NR + n]);
                  svfloat64_t vc3 = svld1_f64(pn, &C_c[(m + 2) * NR + n]);
                  svfloat64_t vc4 = svld1_f64(pn, &C_c[(m + 3) * NR + n]);

                  // svfloat64_t vc1 = svdup_f64(0.0);
                  // svfloat64_t vc2 = svdup_f64(0.0);
                  // svfloat64_t vc3 = svdup_f64(0.0);
                  // svfloat64_t vc4 = svdup_f64(0.0);

                  // svfloat64_t vc1, vc2;
                  // svfloat64_t vc3, vc4;

                  //Step 3: Add and Outproduct the Ar's column vector and Br's row vector 
                  // double sa = A_c[kr * MR + m];

                  // vc = svmla_n_f64_m(pn, vc, vb, sa);

                  vc1 = svmla_lane_f64(vc1, vb, va1, 0);
                  vc2 = svmla_lane_f64(vc2, vb, va1, 1);

                  vc3 = svmla_lane_f64(vc3, vb, va2, 0);
                  vc4 = svmla_lane_f64(vc4, vb, va2, 1);

                  // vc3 = svmla_lane_f64(vc3, vb, va3, 0);
                  // vc3 = svmla_lane_f64(vc3, vb, va3, 1);

                  // vc4 = svmla_lane_f64(vc4, vb, va4, 0);
                  // vc4 = svmla_lane_f64(vc4, vb, va4, 1);

                  
                  // Step 4: Store the vector in
                  // svst1_f64(pn, &C[(index_m) * N + index_n], vc1);
                  // svst1_f64(pn, &C[(index_m + 1) * N + index_n], vc2);
                  // svst1_f64(pn, &C[(index_m + 2) * N + index_n], vc3);
                  // svst1_f64(pn, &C[(index_m + 3) * N + index_n], vc4);
                  svst1_f64(pn, &C_c[(m) * NR + n], vc1);
                  svst1_f64(pn, &C_c[(m + 1) * NR + n], vc2);
                  svst1_f64(pn, &C_c[(m + 2) * NR + n], vc3);
                  svst1_f64(pn, &C_c[(m + 3) * NR + n], vc4);
                }
              }
            }

            for (int m = 0; m < MR && m + mr + mc < M; m ++){
              for (int n = 0; n < NR && n + nr + nc < M; n ++) {
                 C[(m + mr + mc) * N + n + nr + nc] = C_c[m * NR + n];
              }
            }

        }

        }

      }
    }
  }
  
}


void gemm_test(const double *A, const double *B, double *C, const int M, const int K, const int N, const int algorithm, const int ntest) {
  int alpha = 1, beta = 0;
  
  double elapse_time_all = 0;


  for (int i = 0; i < ntest; i++) {
    for (int j = 0; j < M * N; j++) {
      C[j] = 0.0f;
    }
    double start_time = timestamp();
    if (algorithm == 0) {
      kml_dgemm(A, B, C, M, K, N);
    } else {
      sve_dgemm_v3(M, N, K, alpha, A, B, beta, C);
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
      printf("SGEMM: SVE256 ; ");
  }
  // printf("Matrix Size: (%d,%d)x(%d,%d): Theoretical Performance %7.2lf GFlops \n", M,K,K,N, gflops_theory);

  printf("Matrix Size: (%dx%d)X(%dx%d) ; Elapse time (ms): %lf ; GFlops: %7.2lf \n", M,K,K,N, elapse_time * 1000, gflops);
 
}

void ans_verify(const double *ans, const double *C, const int M, const int N) {
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

  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < N; j++) {
  //       printf("%lf ", C[i * N + j]);
  //   }
  //   printf("\n");
  // }

  gemm_test(A, B, ans, M, K, N, 0, LOOP_NUM);
  gemm_test(A, B, C, M, K, N, 1, LOOP_NUM);

  
  ans_verify(ans, C, M, N);

// #ifdef _DEBUG
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < N; j++) {
  //       printf("%lf ", ans[i * N + j]);
  //   }
  //   printf("\n");
  // }

  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < N; j++) {
  //       printf("%lf ", C[i * N + j]);
  //   }
  //   printf("\n");
  // }
// #endif

  free(A);
  free(B);
  free(C);
  free(ans);

  return 0;
}