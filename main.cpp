#include "./ETL/ETL.h"
#include <iostream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;

int main(int argc, char *argv[])
{
    ETL etl(argv[1], argv[2], argv[3]);

    vector<vector<string>> dataset = etl.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);
    Eigen::MatrixXd norm = etl.Normalize(dataMat, true);

    cout << dataMat << endl;

    return EXIT_SUCCESS;
}
