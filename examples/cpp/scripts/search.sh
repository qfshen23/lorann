cd ..
g++ -o ./search ./search.cpp -I ../../lorann/ -I /usr/include/eigen3 -O3 -fopenmp -mavx -msse -std=c++17 -march=native
C=4096
# datasets=('gist' 'deep1M' 'sift')

datasets=('sift')

for data in "${datasets[@]}"
do  
    
    echo "Indexing - ${data}"

    data_path=/data/vector_datasets/${data}
    index_path=/data/tmp/lorann/${data}

    if [ ! -d "$index_path" ]; then 
        mkdir -p "$index_path"
    fi

    rank=32

    data_file="${data_path}/${data}_base.fvecs"
    index_file="${index_path}/${data}_rank${rank}_C${C}.lorann"
    groundtruth_file="${data_path}/${data}_groundtruth.ivecs"
    query_file="${data_path}/${data}_query.fvecs"
    result_file="./${data}_rank${rank}_C${C}_result.txt"

    echo $index_file

    ./search -n $data_file -i $index_file -a $rank -q $query_file -g $groundtruth_file -c $C -r $result_file
done