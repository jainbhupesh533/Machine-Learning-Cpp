#include "ETL.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

using namespace std;

vector<vector<string>> ETL::readCSV()
{
    ifstream file(dataset);
    vector<vector<string>> dataString;

    string line = "";

    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimiter));
        dataString.push_back(vec);
    }
    file.close();
    return dataString;
}

Eigen::MatrixXd ETL::CSVtoEigen(vector<vector<string>> dataset, int rows, int cols)
{
    if (header == true)
    {
        rows = rows - 1;
    }
    Eigen::MatrixXd mat(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; ++j)
        {
            mat(j, i) = atof(dataset[i][j].c_str());
        }
    }
    return mat.transpose();
}

auto ETL::Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean())
{
    return data.colwise().mean();
}

auto ETL::Std(Eigen::MatrixXd data) -> decltype(((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt())
{
    return ((data.array().square().colwise().sum()) / (data.rows() - 1)).sqrt();
}

Eigen::MatrixXd ETL::Normalize(Eigen::MatrixXd data, bool normalizeTarget)
{
    Eigen::MatrixXd dataNorm;
    if (normalizeTarget == true)
    {
        dataNorm = data;
    }
    else
    {
        dataNorm = data.leftCols(data.cols() - 1);
    }
    auto mean_data = Mean(dataNorm);
    Eigen::MatrixXd scaled_data = dataNorm.rowwise() - mean_data;
    auto std_data = Std(scaled_data);
    Eigen::MatrixXd normalized_data = scaled_data.array().rowwise() / std_data;
    if (normalizeTarget == false)
    {
        normalized_data.conservativeResize(normalized_data.rows(), normalized_data.cols() + 1);
        normalized_data.col(normalized_data.cols() - 1) = data.rightCols(1);
    }
    return normalized_data;
}