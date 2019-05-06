/*
 * Lasso solver
 */
#include <Eigen/Dense>

Eigen::VectorXd lasso(Eigen::MatrixXd& A,
          Eigen::VectorXd& y,
          double lambda,
          bool intercept,
          int iter_max);

Eigen::VectorXd async_lasso(Eigen::MatrixXd& A,
          Eigen::VectorXd& y,
          double lambda,
          double stepsize,
          bool intercept,
          int batch_size,
          int epoch);
