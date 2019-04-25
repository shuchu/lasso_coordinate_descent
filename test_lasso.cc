#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Dense>

#include "lasso_io.h"
#include "lasso.h"

void usage(char* argv[]){
    printf("%s [sample] [label] [lambda]\n", argv[0]);
}


int main(int argc, char* argv[])
{
    if (argc != 4){
        usage(argv);
        return 1;
    }
    
    char* A_file_name = argv[1];
    char* y_file_name = argv[2];
    double lambda = atof(argv[3]);
    
    Eigen::MatrixXd mat_A = loadMtxToMatrix(A_file_name);
    Eigen::MatrixXd mat_y = loadMtxToMatrix(y_file_name);

    //map mat_y into a vector
    Eigen::VectorXd vec_y(Eigen::Map<Eigen::VectorXd>(mat_y.data(), mat_y.cols()*mat_y.rows()));

    //debug
    std::cout << "test: " << mat_A(10,10) << std::endl;
    std::cout << mat_y(10) << std::endl;

    bool intercept = false;
    int iter_max = 1000;

    Eigen::VectorXd x = lasso(mat_A, vec_y, lambda, intercept,iter_max); 

    std::cout << x << std::endl;

    return 0;
}
