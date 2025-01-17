#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <getopt.h>
#include <ctime>
#include <chrono>
#include "utilss.h"
#include "lorann.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixInt;

const int MAXK = 100;

RowMatrix load_vectors(std::string data_file_path){
    int n = 0;
    int d = 0;
    std::cout << "Reading: " << data_file_path << std::endl;
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&d, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    n = (size_t)(fsize / (sizeof(float) * d + 4));

    RowMatrix ret(n, d);

    in.seekg(0, std::ios::beg);

    float* buffer = new float[d];

    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(buffer), d * sizeof(float));
        for(int j = 0;j < d; j++){
            ret(i, j) = buffer[j];
        }
    }
    in.close();
    delete[] buffer;
    return ret;
}

RowMatrixInt load_vectors_int(std::string data_file_path) {
    int d = 0;
    int n = 0;
    std::cout << "Reading: " << data_file_path << std::endl;
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&d, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    n = (size_t)(fsize / (sizeof(float) * d + 4));

    RowMatrixInt ret(n, d);

    in.seekg(0, std::ios::beg);

    int* buffer = new int[d];

    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(buffer), d * sizeof(int));
        for(int j = 0;j < d; j++){
            ret(i, j) = buffer[j];
        }
    }
    in.close();
    delete[] buffer;
    return ret;
}

using namespace std;

void test(Lorann::Lorann<Lorann::SQ4Quantizer> &index, const RowMatrix &Q, const RowMatrixInt &G, int k) {

    float sys_t, usr_t, usr_t_sum = 0, total_time=0, search_time=0;
    struct rusage run_start, run_end;  

    vector<int> nprobes;
    nprobes.push_back(10);
    nprobes.push_back(20);
    nprobes.push_back(50);
    nprobes.push_back(80);
    nprobes.push_back(100);
    nprobes.push_back(200);

    
    for (int i = 0; i < nprobes.size(); i++) {
        int nprobe = nprobes[i];
        total_time=0;
        int correct = 0;

        for(int j = 0; j < Q.rows(); j++) {
            Eigen::VectorXf query = Q.row(j);
            Eigen::VectorXi g = G.row(j);
            Eigen::VectorXi indices(k);

            int points_to_rerank = 1000;

            GetCurTime(&run_start);
            index.search(query.data(), k, nprobe, points_to_rerank, indices.data());
            GetCurTime(&run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;

            std::set<int> gt(g.data(), g.data() + g.size());
            
            for(int kk = 0; kk < k; kk++) {
                if(gt.find(indices(kk)) != gt.end()) {
                    correct++;
                }
            }
        }

        float time_us_per_query = total_time / Q.rows();
        float recall = 1.0f * correct / (Q.rows() * k);
        
        // (Search Parameter, Recall, Average Time/Query(us), Total Dimensionality)
        cout << "------------------------------------------------" << endl;
        cout << "nprobe = " << nprobe << " k = " << k  << endl;
        cout << "Recall = " << recall * 100.000 << "%\t" << endl;
        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
    }
}

int main(int argc, char * argv[]) {

    const struct option longopts[] = {
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 
        // Query Parameter 
        {"K",                           required_argument, 0, 'k'},
        {"C",                           required_argument, 0, 'c'},
        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        // {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"index_path",                  required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    // getopt error message (off: 0)

    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char index_path[256] = "";

    int rank = 32;
    int subk = 100;
    int C = 4096;

    while(iarg != -1) {
        iarg = getopt_long(argc, argv, "q:g:r:n:k:p:a:c:i:", longopts, &ind);
        switch (iarg){  
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
            case 'a':
                if(optarg)rank = atoi(optarg);
                break;
            case 'k':
                if(optarg)subk = atoi(optarg);
                break;
            case 'q':   
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':   
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':   
                if(optarg)strcpy(result_path, optarg);
                break;  
            case 'c':
                if(optarg)C = atoi(optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
        }
    }

    subk=100;
    
    RowMatrix X = load_vectors(dataset);    
    RowMatrix Q = load_vectors(query_path);
    RowMatrixInt G = load_vectors_int(groundtruth_path);

    freopen(result_path, "a", stdout);

    const int n_clusters = C;
    const int global_dim = X.cols();
    
    const int train_size = 5;
    const bool euclidean = true;

    Lorann::Lorann<Lorann::SQ4Quantizer> index(X.data(), X.rows(), X.cols(), n_clusters, global_dim,
                                             rank, train_size, euclidean);

    std::ifstream input_file(index_path, std::ios::binary);
    if (!input_file) {
        throw std::runtime_error("Failed to open index file!");
    }
    cereal::BinaryInputArchive input_archive(input_file);
    input_archive(index);
    std::cerr << "Index loaded successfully!" << std::endl;

    std::cerr << "Testing the index..." << std::endl;
    test(index, Q, G, subk);

    return 0;
}