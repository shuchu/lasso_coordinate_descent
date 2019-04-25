/*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY offsets accordingly when reading and writing 
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <Eigen/Dense>

#include "mmio.h"
#include "lasso_io.h"

// load matrix from .mtx file into a MatrixXd object
Eigen::MatrixXd loadMtxToMatrix(char* file_name)
{
    std::FILE *f;
    int ret_code;
    MM_typecode matcode;
    int M, N, nz;

    if ((f = fopen(file_name, "r")) == 0)
        exit(1);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    // This is how one can screen matrix types if their application 
    // only supports a subset of the Matrix Market data types.    

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    // find out size of sparse matrix .... 
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);
    
    // create a MatrixXd object
    Eigen::MatrixXd mat = Eigen::MatrixXd(M,N);
    
    // NOTE: when reading in doubles, ANSI C requires the use of the "l"  
    //   specifier as in "%lg", "%lf", "%le", otherwise errors will occur 
    //  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            
    int m_i(0), m_j(0);
    double val(0);
    for (int i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &m_i, &m_j, &val);
        mat(m_i-1, m_j-1) = val;
    }

    if (f !=stdin) fclose(f);
    
    // show loaded matrix shape
    std::cout << "matrix size: ("<< M << "," << N << ")." << std::endl;

    return mat;
}

