#pragma once
#include <string>
#include <Eigen/Dense>


#ifndef _EPS_
#define _EPS_  2.2204e-16
#endif

Eigen::MatrixXd loadMtxToMatrix(char* file_name);

