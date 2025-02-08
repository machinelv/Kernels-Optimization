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

#include "arm_sme.h"
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



//Cache Partition
#define MC 128
#define KC 128
#define NC 128

//Register Partition
#define MR 64
#define NR 64

__attribute__((noinline))__arm_new("za") void dgemm_SME_v0(const double *A, const double *B, double *C,
                                                        unsigned long M, unsigned long K, unsigned long N, double alpha) __arm_streaming{
  uint64_t vscale;
  svbool_t pm, pn, pk;
  svfloat64_t src1, src2;
  vscale = svcntd();
  
  for (size_t i=0; i<M; i+=vscale){
      pm=svwhilelt_b64_u32(i,M);
      for (size_t j=0; j<N; j+=vscale){
          pn=svwhilelt_b64_u32(j,N);
          svzero_mask_za(0);
        for (size_t k=0; k<K; k+=vscale){
              pk=svwhilelt_b64_u32(k,K);
              for (size_t t=0; t<vscale; t++){
                  if (i + t == M)
                      break;
                  svld1_hor_za64(1, t, pk, A + (i+t) * K + k);
              }
              for (size_t t=0; t < vscale; t++){
                  if (k+t==K)
                      break;
                  src1=svread_ver_za64_f64_m(src1, pm, 1, t);
                  src2=svld1_f64(pn,B+(k+t)*N+j);
                  src2=svmul_n_f64_m(pn,src2,alpha);
                  svmopa_za64_f64_m(0,pm,pn,src1,src2);
              }
        }
      }
  }
}

#define MASK_ZACC ((1 << 0) | (1 << 4))
#define ZACC 0



__attribute__((noinline))__arm_new("za") void dgemm_SME_v1(const double *A, const double *B, double *C,
                                                        unsigned long M, unsigned long K, unsigned long N, double alpha) __arm_streaming{
  uint64_t vscale = svcntw();
  svbool_t pm, pn, pk;
  svfloat64_t src1, src2;
  // Divide C into multiple [vscale x vscale] tiles, with the (i, j) pair in
  // each iteration indicating the top-left coordinate of a tile in C.
  for (size_t i = 0; i < M; i += vscale) {
    pm = svwhilelt_b64_u32(i, M); // predicate for rows of matrixes A and C
    for (size_t j = 0; j < N; j += vscale) {
      pn = svwhilelt_b64_u32(j, N); // predicate for columns of matrixes B and C
      svzero_mask_za(MASK_ZACC);
      // The matrix multiplication of two [vscale x vscale] tiles is equal to
      // the sum of outer products of each column of the first tile and each row
      // of the second.
      for (size_t k = 0; k < K; k += vscale) {
        pk = svwhilelt_b64_u32(k, K); // predicate for columns of A and rows of B
        // Multiply columns of the [vscale x vscale] tile starting at A[k][i]
        // with rows of the tile starting at B[k][j].
        for (size_t t = 0; t < vscale; t++) {
          // Tiles along the right hand side of matrix A will only have (K %
          // vscale) columns. Tiles along the bottom of matrix B will only have
          // (K % vscale) rows. Exit early if we have reached the limit.
          if (k + t == K)
            break;
          // pm will prevent loading more column-wise elements than available
          // when loading the consecutive elements starting at A[k + t][i].
          src1 = svld1_f64(pm, A + i * K + (k + t), );
          // pn will prevent extracting more row-wise elements than available
          // when loading the consecutive elements starting at B[k + t][j].
          src2 = svld1_f64(pn, B + (k + t) * N + j);
          // Multiply with alpha.
          src2 = svmul_n_f64_m(pn, src2, alpha);
          // Accumulate the outer product of one column from a tile of A and
          // one row from a tile of B.
          svmopa_za64_f64_m(ZACC, pm, pn, src1, src2);
        }
      }
      // Copy the content of the accumulator tile, row-wise, into the
      // corresponding tile of C.
      for (size_t t = 0; t < vscale; t++) {
        // Tiles along the bottom of matrix C will only have (M % vscale) rows.
        // Exit early if we have stored all rows available.
        if (i + t == M)
        break;
        // pn will prevent storing more row-wise elements than necessary when
        // storing to consecutive elements starting at C[i][j].
        svst1_hor_za64(ZACC, t, pn, C + (i + t) * N + j);
      }
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

__attribute__((noinline))__arm_new("za") void dgemm_SME_v2(const double *A, const double *B, double *C,
                                                        unsigned long M, unsigned long K, unsigned long N, double alpha) __arm_streaming{
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

            svzero_mask_za(MASK_ZACC);

            for (int kr = 0; kr < KC; kr ++) {
              //MR must be equal to NR
              for (int m = 0; m < MR; m += vl) {
                for (int n = 0; n < NR; n += vl) {

                  int index_m = m + mr + mc;
                  int index_n = n + nr + nc;
                  int index_k = kr + kc;

                  svbool_t pn = svptrue_b64();
                  svfloat64_t va = svld1_f64(pn, &A_c[kr * MR + m]);
                  svfloat64_t vb = svld1_f64(pn, &B_c[kr * NR + n]);
                  // svfloat64_t vb = svld1_f64(pn, &B[index_k * N + index_n]);


                  // Multiply with alpha.
                  vb = svmul_n_f64_m(pn, vb, alpha);
                  // Accumulate the outer product of one column from a tile of A and
                  // one row from a tile of B.
                  svmopa_za64_f64_m(ZACC, pm, pn, va, vb);
                  
                  // Step 4: Store the vector in
                  
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
      dgemm_SME_v1(A, B, C, M, K, N, 1);
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