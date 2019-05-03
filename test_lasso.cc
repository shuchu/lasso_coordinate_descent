#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Dense>

#include "lasso_io.h"
#include "lasso.h"

void usage(char* argv[]){
    printf("%s [sample] [label] [lambda] [max_ite]\n", argv[0]);
}


int main(int argc, char* argv[])
{
    if (argc != 5){
        usage(argv);
        return 1;
    }
    
    char* A_file_name = argv[1];
    char* y_file_name = argv[2];
    double lambda = atof(argv[3]);
    int max_iter = atoi(argv[4]);
    
    Eigen::MatrixXd mat_A = loadMtxToMatrix(A_file_name);
    Eigen::MatrixXd mat_y = loadMtxToMatrix(y_file_name);

    // center and normalize
    Eigen::VectorXd mat_A_mean = mat_A.colwise().mean();
    mat_A.rowwise() -= mat_A_mean.transpose();

    // normalize columns of matrix A
    Eigen::VectorXd A_norm = mat_A.colwise().norm();
    for (int i = 0; i < A_norm.size(); i++) {
        if (A_norm(i) == 0.0)
            A_norm(i) = 1.0;
        else 
            A_norm(i) = 1.0 / A_norm(i);
    }
    
    mat_A = mat_A * (A_norm.asDiagonal());

    //map mat_y into a vector
    Eigen::VectorXd vec_y(Eigen::Map<Eigen::VectorXd>(mat_y.data(), mat_y.cols()*mat_y.rows()));

    bool intercept = false 

    Eigen::VectorXd x = lasso(mat_A, vec_y, lambda, intercept,max_iter); 

    std::cout << x << std::endl;

    return 0;
}
