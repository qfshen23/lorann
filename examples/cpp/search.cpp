#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <getopt.h>

#include "lorann.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;

template <typename T>
RowMatrix load_vectors(std::string data_file_path) {
    int n = 0;
    int d = 0;
    printf("Reading: %s\n",data_file_path);
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&d, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    n = (size_t)(fsize / (sizeof(T) * d + 4));

    RowMatrix ret(n, d);

    in.seekg(0, std::ios::beg);

    T* buffer = new T[d];

    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(buffer), d * sizeof(T));
        for(int j = 0;j < d; j++){
            ret(i, j) = buffer[j];
        }
    }
    in.close();
    delete[] buffer;
    return ret;
}

using namespace std;

int n = 0;
int d = 0;

const int MAXK = 100;

void test(const RowMatrix<float> &Q, const RowMatrix<unsigned> &G, const IVF &ivf, int k, float* hashed_queries) {
    float sys_t, usr_t, usr_t_sum = 0, total_time=0, search_time=0;
    struct rusage run_start, run_end;   

    vector<int> nprobes;
    nprobes.push_back(10);
    nprobes.push_back(20);
    nprobes.push_back(30);
    // nprobes.push_back(50);
    // nprobes.push_back(40);
    // nprobes.push_back(80);
    nprobes.push_back(100);
    nprobes.push_back(200);
    nprobes.push_back(300);
    
    for(auto nprobe:nprobes){
        total_time=0;
        adsampling::clear();
        int correct = 0;

        for(int i=0;i<Q.n;i++){
            GetCurTime(&run_start);
            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, hashed_queries + i * ivf.hash_len, k, nprobe);
            GetCurTime(&run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
            // Recall
            while(KNNs.empty() == false){
                int id = KNNs.top().second;
                KNNs.pop();
                for(int j=0;j<k;j++)
                    if(id == G.data[i * G.d + j]) correct ++;
            }
        }
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);
        
        // (Search Parameter, Recall, Average Time/Query(us), Total Dimensionality)
        cout << "------------------------------------------------" << endl;
        cout << "nprobe = " << nprobe << " k = " << k  << " prop= " << ivf.candidate_prop << "%" <<  endl;
        cout << "Recall = " << recall * 100.000 << "%\t" << endl;
        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        cout << "time1 = " << adsampling::time1 << " time2 = " << adsampling::time2 << endl;
    }
}

int main(int argc, char * argv[]) {

    const struct option longopts[] = {
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 
        // Query Parameter 
        {"K",                           required_argument, 0, 'k'},
        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    // getopt error message (off: 0)

    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char index_path[256] = "";

    int subk = 100;

    while(iarg != -1) {
        iarg = getopt_long(argc, argv, "q:g:r:n:k:p:i:", longopts, &ind);
        switch (iarg){
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
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;

        }
    }
    
    RowMatrix Q = load_vectors<float>(query_path);
    RowMatrix G = load_vectors<unsigned>(groundtruth_path);
    freopen(result_path, "a", stdout);
    Lorann::Lorann<Lorann::SQ4Quantizer> index;
    index.load(index_path);
    test(Q, G, ivf, subk);
    return 0;
}



int main() {
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
    int C=4096;

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
        }
    }
    
    RowMatrix X = load_vectors<float>(dataset);
    RowMatrix Q = load_vectors<float>(query_path);
    RowMatrix G = load_vectors<unsigned>(groundtruth_path);

    const int n_clusters = C;
    const int global_dim = d;
    
    const int train_size = 5;
    const bool euclidean = true;

    Lorann::Lorann<Lorann::SQ4Quantizer> index(X.data(), X.rows(), X.cols(), n_clusters, global_dim,
                                             rank, train_size, euclidean);
    
    std::cout << "Saving the index to disk..." << std::endl;
    std::ofstream output_file(index_path, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(output_file);
    output_archive(index);    
    
    std::cout << "Reading the index from disk..." << std::endl;
    
    return 0;
}