#include <iostream>
#include <vector>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int main () {
    return 0;
}

extern "C" {
    void matmul(int transA, int transB, int A_height, int A_width, int B_height, int B_width, double* A, double* B, double* output) {
        bool tA = transA == 1;
        bool tB = transB == 1;
        int A_h = tA ? A_width  : A_height;
        int A_w  = tA ? A_height : A_width;
        int B_h = tB ? B_width  : B_height;
        int B_w  = tB ? B_height : B_width;
        int M = A_h;
        int N = B_w;
        int K = A_w;
        int lda = tA ? M : K;
        int ldb = tB ? K : N;

        cblas_dgemm(CblasRowMajor, 
            tA ? CblasTrans : CblasNoTrans,
            tB ? CblasTrans : CblasNoTrans, 
            M, N, K, 1, 
            A, lda, 
            B, ldb, 
            0.0, 
            output, N);
    }
}

