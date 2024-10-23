#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "lorann.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;

RowMatrix load_vectors() {
  std::ios::sync_with_stdio(false);

  std::ifstream fin("wiki-news-300d-1M.vec");
  if (!fin.is_open()) {
    throw std::runtime_error(
        "Could not open wiki-news-300d-1M.vec. Run `make prepare-data` first.");
  }

  std::string line;
  std::getline(fin, line);
  std::istringstream header(line);
  int n, d;
  header >> n >> d;

  RowMatrix ret(999994, 300);

  int i = 0;
  while (std::getline(fin, line)) {
    std::istringstream iss(line);
    std::string token;
    iss >> token;

    int j = 0;
    float value;
    while (iss >> value) {
      ret(i, j) = value;
      ++j;
    }
    ++i;
  }

  return ret;
}

int main() {
  std::cout << "Loading data..." << std::endl;
  RowMatrix X = load_vectors();
  RowMatrix Q = X.topRows(1000);

  const int k = 10;

  const int n_clusters = 1024;
  const int global_dim = 256;
  const int rank = 32;
  const int train_size = 5;
  const bool euclidean = true;

  const int clusters_to_search = 64;
  const int points_to_rerank = 2000;

  std::cout << "Building the index..." << std::endl;
  Lorann::Lorann<Lorann::SQ4Quantizer> index(X.data(), X.rows(), X.cols(), n_clusters, global_dim,
                                             rank, train_size, euclidean);
  index.build();
  
  Eigen::VectorXi indices(k), indices_exact(k);

  std::cout << "Querying the index using exact search..." << std::endl;
  index.exact_search(Q.row(0).data(), k, indices_exact.data());
  std::cout << indices_exact.transpose() << std::endl;

  std::cout << "Querying the index using approximate search..." << std::endl;
  index.search(Q.row(0).data(), k, clusters_to_search, points_to_rerank, indices.data());
  std::cout << indices.transpose() << std::endl;

  std::cout << "Saving the index to disk..." << std::endl;
  std::ofstream output_file("index.bin", std::ios::binary);
  cereal::BinaryOutputArchive output_archive(output_file);
  output_archive(index);

  return 0;
}
