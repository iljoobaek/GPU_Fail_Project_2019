set -e

inputs=('kmeans01.dat' 'kmeans02.dat' 'kmeans03.dat' 'kmeans04.dat')

mkdir -p profiles

make cuda

echo "--------------------------------------------------------------------------------"
uptime
echo "--------------------------------------------------------------------------------"

# Add quotes around ${input} so that spaces in the filename don't break things

cudaTotalTime=0

for input in ${inputs[@]}; do
    cudaFileTime=0

    for k in 2 4 8 16 32 64 128; do
        cudaTime=$(./cuda_main -o -n $k -i data/${input} | grep 'Computation' | awk '{print $4}')
        mv data/${input}.cluster_centres data/${input}-$k.cluster_centres
        mv data/${input}.membership data/${input}-$k.membership


        echo "k = $(printf "%3d" $k)  cudaTime = ${cudaTime}s"

        # aggregate file times
        cudaFileTime=$(echo "${cudaFileTime} + ${cudaTime}" | bc)
    done

    # aggregate total times
    cudaTotalTime=$(echo "${cudaTotalTime} + ${cudaFileTime}" | bc)
    echo "--------------------------------------------------------------------------------"
done

