#include <iostream>
#include <string>
#include <cstdlib>
#include <Eigen/Dense>

#include "lasso_io.h"
#include "lasso.h"


int main(int argc, char* argv[])
{
    std::string _usage("test_lasso [sample] [label] [lambda]\n");
    if (argc != 3){
        std::cerr << _usage << std::endl;
        return 1;
    }
    
    char* A_file_name = argv[0];
    char* y_file_name = argv[1];
    double lambda = atof(argv[2]);
    
    Eigen::MatrixXd mat_A;
    Eigen::MatrixXd mat_y;

    loadMtxToMatrix(A_file_name, mat_A);
    loadMtxToMatrix(y_file_name, mat_y);

    //map mat_y into a vector
    Eigen::VectorXd vec_y(Eigen::Map<Eigen::VectorXd>(mat_y.data(), mat_y.cols()*mat_y.rows()));

    bool intercept = false;
    int iter_max = 1000;

    Eigen::VectorXd x = lasso(mat_A, vec_y, lambda, intercept,iter_max); 

    std::cout << x << std::endl;

    return 0;
}
