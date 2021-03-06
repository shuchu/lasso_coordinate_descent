#include <iostream>
#include <string>
#include <cstdlib>

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
        for (int j=0; j < A.rows(); j++)
            A(j, A.cols()-1) = 1.0;
    }

    // initialize solution vector x, with 0.
    Eigen::VectorXd x(A.cols());
    x.setZero();  
    if (intercept) {
        x(x.size()-1) = y.sum() / y.size();
    }

    // main loop for descent
    for(int ite=0; ite < iter_max; ite++) {
        //coordinate descent
        //pre-calc r = y-AX 
        //following the trick in lecture slides of "coordinate descent"
        //from Geoff Gordon & Ryan Tibshirani at P10.
        //Eigen::VectorXd r = y - A*x;     

        for (int idx = 0; idx < x.size()-1; idx++) {
            Eigen::VectorXd r = y - A*x;     
            double rho = A.col(idx).transpose()*r + x(idx);
            x(idx) = softThresholdingOperator(rho, lambda);
        }

        if (intercept){
            x(x.size()-1) = (y-A*x).sum()/A.rows();
        }       

        //info
        Eigen::VectorXd r = y - A*x;     
        std::cout <<"ite: " << ite <<" residual: "<< r.norm() << std::endl;
    }
        
    return x;
}


Eigen::VectorXd async_lasso(Eigen::MatrixXd& A,
          Eigen::VectorXd& y,
          double lambda,
          double stepsize,
          bool intercept,
          int batch_size,
          int epoch)
{
    //Configure rand() seed
    //std::srand(100);
    
    // add one columne to A if intercept is set to true
    if (intercept){
        A.conservativeResize(A.rows(), A.cols()+1); 
        for (int j=0; j < A.rows(); j++)
            A(j, A.cols()-1) = 1.0;
    }

    // initialize solution vector x, with 0.
    Eigen::VectorXd x(A.cols());
    for (int i = 0; i < x.size(); i++)
        x(i) = 1.0;
    if (intercept) {
        x(x.size()-1) = y.sum() / y.size();
    }


    // pre-compute the (1) A_T*A, (2) b^T*A
    Eigen::MatrixXd AtA = A.transpose()*A;
    AtA /= A.rows();
    Eigen::MatrixXd btA = y.transpose()*A;
    btA /= A.rows();
    
    // main loop for descent
    int cnt = 0;
    int x_org_size = x.size();
    double res = -1.0;
    if (intercept) x_org_size = x.size()-1;
    while(epoch > cnt++) {
        //coordinate descent
        for (int i = 0; i< batch_size; i++) {
            int idx = rand()%x_org_size;
            double grad_f_x_j = AtA.col(idx).sum()*x(idx) - btA(idx);
            //calcualte the stepsize
            double update = x(idx) - stepsize*grad_f_x_j;
            //now update
            x(idx) = softThresholdingOperator(update, stepsize*lambda);
        }

        if (intercept){
            x(x.size()-1) = 0.0;
            x(x.size()-1) = (y-A*x).sum()/A.rows();
        }       

        //info
        Eigen::VectorXd r = y - A*x;     
        r /= A.rows();
        
        if (fabs(r.norm() - res) < 1e-5) {
            break;
        }else
            res = r.norm();
        
        //debug
        std::cout <<"epoch: " << cnt <<" residual: "<< r.norm() << " x norm: "<< x.norm() << " intercept: " << x(x.size()-1) << std::endl;
    }
        
    return x;
}
