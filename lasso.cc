#include <iostream>
#include <string>

#include "lasso_io.h"
#include "lasso.h"


inline double softThresholdingOperator(double rho, double lambda){
    if (rho < -lambda) return lambda + rho;
    else if (rho > lambda) return rho - lambda;
    else return 0;
} 

// solve the lasso problem by using coordinate descent
// ||Ax - y ||^2 + lambda*|x|
//
// Assumptions:
//  1. A: each row is a sample, and each column is a feature.
//  2. number of row of A is equal to the length of vector y
Eigen::VectorXd lasso(Eigen::MatrixXd& A,
          Eigen::VectorXd& y,
          double lambda,
          bool intercept,
          int iter_max)
{
    // add one columne to A if intercept is set to true
    if (intercept){
        A.conservativeResize(A.rows(), A.cols()+1); 
        A.col(A.cols()-1).setIdentity();  
    }
    
    // initialize solution vector x, with 0.
    Eigen::VectorXd x(A.cols());
    x.setZero();  

    // normalize columns of matrix A
    Eigen::VectorXd A_norm = A.colwise().squaredNorm();
    for (int i = 0; i < A_norm.size(); i++) {
        if (A_norm(i) == 0.0)
            A_norm(i) = 1.0;
        else 
            A_norm(i) = 1.0 / A_norm(i);
    }
    
    A = A * (A_norm.asDiagonal());

    // main loop for descent
    for (int iter = 0; iter < iter_max; iter++) {
        //coordinate descent
       
        //pre-calc r = y-AX 
        //following the trick in lecture slides of "coordinate descent"
        //from Geoff Gordon & Ryan Tibshirani at P10.
        Eigen::VectorXd r = y - A*x;     

        for (int idx = 0; idx < x.size(); idx++) {
            double rho = A.col(idx).transpose() * r;
            if (intercept){
                if (idx == x.size() -1) x(idx) = rho;
                else x(idx) = softThresholdingOperator(rho, lambda);
            }else {
                x(idx) = softThresholdingOperator(rho, lambda);
            }       
        } 
    }
        
    return x;
}
