/*
 * Lasso solver
 */
#include <Eigen/Dense>

Eigen::VectorXd lasso(Eigen::MatrixXd& A,
          Eigen::VectorXd& y,
          double lambda,
          bool intercept,
          int iter_max);


