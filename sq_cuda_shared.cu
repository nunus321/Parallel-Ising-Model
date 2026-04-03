#include <iostream>
#include <math.h>
#include <bits/stdc++.h>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <stdio.h> 
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <curand_kernel.h> 
#include <list>
#include <filesystem>
#include <cassert>
#include <numeric>
#include <algorithm>
using namespace std;
std::random_device rd;
std::mt19937 mt;
std::uniform_int_distribution<int> unifIntDistr0to1(0, 1);
std::uniform_real_distribution<double> unifReDistr0to1(0, 1.0);
std::normal_distribution<double> normDist(0.0,1.0);

//const double PI=3.141592653287979;
#define tile_w 16
#define tile_h 16
#define ghost 1
static inline void cudaCheck(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

class Atoms{
public:
    
    vector<int> z;
    size_t no_atoms;
    vector<vector<size_t>> neighbours;
    double J;
    double K;
    double D;
    double B;
    double kbT;

    Atoms(vector<int> z, vector<vector<size_t>> neighbours, size_t no_atoms,double J, double K, double D, double B, double kbT)
    :z{z}, no_atoms{no_atoms}, neighbours{neighbours} , J{J}, K{K}, D{D}, B{B}, kbT{kbT}
    {}
};

class System{
public:
    const size_t size;
    double E=0;
    Atoms atoms;
    System(size_t size, double J, double K, double D, double B, double kbT, vector<int> z, vector<vector<size_t>> neighbours)
    :size{size}, atoms(z , neighbours, size*size, J, K, D, B, kbT)
    {}
    


};


class Sim_Configuration {
public:
    size_t steps = 10000;     // number of steps
    size_t size = 100;      // lattice size
    size_t data_period = 100; // how often to save states
    double J=1;
    double K=0;
    double kbT=0.001;
    double D=0;
    double B=1;
    string init_config="";
    std::string filename = "Config.dat";   // name of the output file with configuration
    std::string filenameE= "Energy.dat";   // nam of the output file with eenergis
    // system box size. for this code these values are only used for vmd, but in general md codes, period boundary conditions exist

    // simulation configurations: number of step, number of the molecules in the system, 
    // IO frequency and file name
    Sim_Configuration(std::vector <std::string> argument){
        for (size_t i = 1; i<argument.size() ; i += 2){
            std::string arg = argument.at(i);
            if(arg=="-h"){ // Write help
                std::cout << "MC -steps <number of steps> -size <Lattice side>"
                          << " -J <Coupling energy> -D <DMI strength>"
                          << " -B <Magnetic Field> -K <Anisotropy strngth> -kbT <Temperature>"
                          << " -init <initial config type> -fwrite <io frequency>"
                          << " -ofileE <filename Energies> -ofile <filename Snapshots> \n";
                exit(0);
                break;
            } else if(arg=="-steps"){
                steps = std::stoi(argument[i+1]);
            } else if(arg=="-size"){
                size = std::stoi(argument[i+1]);
            } else if(arg=="-fwrite"){
                data_period = std::stoi(argument[i+1]);
            } else if(arg=="-ofile"){
                filename = argument[i+1];
            } else if(arg=="-ofileE"){
                filenameE = argument[i+1];
            } else if(arg=="-J"){
                J = std::stod(argument[i+1]);
            } else if(arg=="-D"){
                D = std::stod(argument[i+1]);
            } else if(arg=="-B"){
                B = std::stod(argument[i+1]);
            } else if(arg=="-K"){
                K = std::stod(argument[i+1]);
            } else if(arg=="-kbT"){
                kbT = std::stod(argument[i+1]);
            } else if(arg=="-init"){
                filename = argument[i+1];
                
                
            } else{
                std::cout << "---> error: the argument type "<<arg<<" is not recognized \n";
            }
        }

        
    }
};

int NewVec(){
    //double p1=unifReDistr0to1(mt)*2-1;
    //double p2=unifReDistr0to1(mt)*2-1;
    int p3=unifIntDistr0to1(mt)*2-1;
    
    return p3;
}

void RandInitCondition(System& sys){
    for(size_t i=0;i<sys.atoms.no_atoms;i++){
        int xyz=NewVec();
        //sys.atoms.x[i]=xyz[0];
        //sys.atoms.y[i]=xyz[1];
        sys.atoms.z[i]=xyz;
        
        
    }
}

void AntiInitCondition(System& sys){
    for(size_t i=0;i<sys.atoms.no_atoms;i++){
        if(i%2==0){
        
        sys.atoms.z[i]=1;
        }else{
        sys.atoms.z[i]=-1;
        }
    }
}

void CircInitCondition(System& sys, int r){
    for(size_t i=0;i<sys.atoms.no_atoms;i++){
        int col = i % sys.size;
        int row = i / sys.size;
        double half=(double)sys.size/2;

        if((pow(col-half,2)+pow(row-half,2))<(r*r)){
        
        sys.atoms.z[i]=-1;
        }else{
        sys.atoms.z[i]=1;
        }
    }
}


System MakeLattice(size_t size, double J, double K, double D, double B, double kbT){
    const size_t no_atoms=size*size;

    vector<int>z(no_atoms);
    vector<vector<size_t>> neighbours(no_atoms, vector<size_t>(4)); //top bottom left right
    
    for(size_t i=0;i<no_atoms;i++){
        if(i>no_atoms-size-1) neighbours[i][0]=i%size; else neighbours[i][0]=i+size;
    }
    for(size_t i=0;i<no_atoms;i++){
        if(i<size) neighbours[i][1]=no_atoms-size+i; else neighbours[i][1]=i-size;
    }
    for(size_t i=0;i<no_atoms;i++){
        if(i%size==0) neighbours[i][2]=i+size-1; else neighbours[i][2]=i-1;
    } 
    for(size_t i=0;i<no_atoms;i++){
        if((i+1)%size==0) neighbours[i][3]=i-size+1; else neighbours[i][3]=i+1;
    }
        
    
    System sys( size,  J, K,  D,  B,  kbT, z, neighbours);
    RandInitCondition(sys);

    
    return sys;
}

// __device__ version of Ecalc: same logic as the CPU counterpart but
// operates on the device spin array (int* dz) instead of System&.
__device__ double Ecalc(int* dz, int spin, int i, int L, double J, double B){
    double E=0;
    int no_atoms=L*L;
    //const int dx[4] = { 0,  0, -1, 1};
    //const int dy[4] = { 1, -1,  0, 0};
    int neighbours[4]={0,0,0,0};
        if(i>no_atoms-L-1) neighbours[0]=i%L; else neighbours[0]=i+L;
        if(i<L) neighbours[1]=no_atoms-L+i; else neighbours[1]=i-L;
        if(i%L==0) neighbours[2]=i+L-1; else neighbours[2]=i-1;
        if((i+1)%L==0) neighbours[3]=i-L+1; else neighbours[3]=i+1;
    for (int k = 0; k < 4; k++) {
        int j = neighbours[k];
        int Sjz = dz[j];

 
        E -= J * spin*Sjz;

    
        //double cross_x = spin[1]*Sjz - spin[2]*Sjy;
        //double cross_y = spin[2]*Sjx - spin[0]*Sjz;
  

        //E -= D * (dx[k]*cross_x + dy[k]*cross_y);

    }
    //E-=K*spin*spin;
    E-=B*spin;
    
    return E;
    
}

// EcalcTot remains a host function (used for initial/final energy on CPU)
double EcalcTot(System& sys, int& spin, size_t& i){
    double E=0;
    Atoms& atoms=sys.atoms;
    size_t size=sys.size;
    size_t no_atoms=sys.atoms.no_atoms;
    //const int dx[4] = { 0,  0, -1, 1};
    //const int dy[4] = { 1, -1,  0, 0};

    int neighbours[4]={0,0,0,0};
    if(i>no_atoms-size-1) neighbours[0]=i%size; else neighbours[0]=i+size;
    if(i<size) neighbours[1]=no_atoms-size+i; else neighbours[1]=i-size;
    if(i%size==0) neighbours[2]=i+size-1; else neighbours[2]=i-1;
    if((i+1)%size==0) neighbours[3]=i-size+1; else neighbours[3]=i+1;
    for (int k : {0, 3}) {
            int j = neighbours[k];
            int Sjz = atoms.z[j];

            E -= atoms.J * spin*Sjz;

            //double cross_x = spin[1]*Sjz - spin[2]*Sjy;
            //double cross_y = spin[2]*Sjx - spin[0]*Sjz;
            //E -= atoms.D * (dx[k]*cross_x + dy[k]*cross_y);
        }


    
    //E-=atoms.K*spin[2]*spin[2];
    E-=atoms.B*spin;
    
    return E;
    
}

// Initialise one curandState per lattice site
__global__ void InitRng(curandState* states, unsigned long long seed, int n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<n) curand_init(tid,(unsigned long long)tid,0,&states[tid]);
}

// Parallel sweep over even-parity sites: (x+y)%2==0
// Parity is based on 2D coordinates — required for correct parallel independence
// actually use one function that takes parity as an input and call that twice
__global__ void SweepAll(int* dz, int L, double J, double B, double kbT,
                           curandState* states, double* dEtot,
                           int parity){
    
    __shared__ int tile[(tile_h +2*ghost)*(tile_w+2*ghost)];
    __shared__ double deltaE;
    
    // thread indices within each block (local indices -- 2D)
    // linear form would be row x num_columns + column, so to invert:
    // column = linear % num_columns
    // row = linear / num_columns
    int t_id = threadIdx.x;
    int tx = t_id % tile_w; // (from 0 to 15)
    int ty = t_id / tile_w; // (from 0 to 15)

    // position of block in 2d grid
    int block_x = blockIdx.x*tile_w; 
    int block_y = blockIdx.y*tile_h; 

    // global coordinates of the center of the block 
    int gx = block_x + tx;
    int gy = block_y + ty;
    int gidx = gy * L +gx; //global index in the dz array which is linear

    int int_k=(ty+ghost)*(tile_w+2*ghost)+(tx+ghost); // thread mapping to interior cell index

    curandState localState = states[gidx]; 
    //double localDeltaE = 0.0;
    if (t_id == 0) {
        deltaE = 0.0;
    }
    if (t_id < tile_w*tile_h) { 
        
        tile[int_k] = dz[gidx];

        if(ty==0){
            int ghost_k_bot=(ty-1+ghost)*(tile_w+2*ghost)+(tx+ghost); //bottom threads mapping to bottom ghost zone
            int ghost_gy=((gy-1)+L)%L; // 
            int ghost_gidx_bot=ghost_gy*L+gx; // 
            tile[ghost_k_bot]=dz[ghost_gidx_bot];
        }
        if(ty==tile_h-1){
            int ghost_k_top=(ty+1+ghost)*(tile_w+2*ghost)+(tx+ghost); // top threads mapping to top ghost zone
            int ghost_gy=((gy+1)+L)%L; //
            int ghost_gidx=ghost_gy*L+gx; // 
            tile[ghost_k_top]=dz[ghost_gidx]; //
        }
        if(tx==0){
            int ghost_k_left=(ty+ghost)*(tile_w+2*ghost)+(tx+ghost-1);
            int ghost_gx=((gx-1)+L)%L; //
            int ghost_gidx_left=gy*L+ghost_gx; // 
            tile[ghost_k_left]=dz[ghost_gidx_left]; //
        }
        if(tx==tile_w-1){
            int ghost_k_right=(ty+ghost)*(tile_w+2*ghost)+(tx+ghost+1);
            int ghost_gx=((gx+1)+L)%L; //
            int ghost_gidx_right=gy*L+ghost_gx; // 
            tile[ghost_k_right]=dz[ghost_gidx_right]; //
        }
            

        __syncthreads(); // sync to make sure that the local grid is build consistently
    
        int x = gx % L;
        int y = gy % L;
        bool should_update = ((x + y) % 2 == parity); //0 for even and 1 for odd
    
        if (should_update && t_id < tile_w*tile_h) {
            int spin = (curand_uniform(&localState) < 0.5f) ? -1 : 1;
            
            int oldspin = tile[int_k];
            
            double E0 = 0, En = 0;
            if (oldspin != spin) {
            // Current state energy
            E0 -= J * oldspin * tile[int_k-1];    // left neighbor
            E0 -= J * oldspin * tile[int_k+1];    // right neighbor
            E0 -= J * oldspin * tile[int_k+tile_w+2*ghost];    // top neighbor
            E0 -= J * oldspin * tile[int_k-(tile_w+2*ghost)];    // bottom neighbor
            E0 -= B * oldspin;
            
            // New state energy
            En -= J * spin * tile[int_k-1];
            En -= J * spin * tile[int_k+1];
            En -= J * spin * tile[int_k+tile_w+2*ghost];
            En -= J * spin * tile[int_k-(tile_w+2*ghost)];
            En -= B * spin;
            
            // Metropolis acceptance
            double p = curand_uniform_double(&localState);
            if (exp(-(En - E0) / kbT) > p) {
                dz[gidx] = spin;
                //tile[ghost+ty][ghost+tx] = spin;
                #ifdef CHECKSUM
                atomicAdd(&deltaE, En - E0);
                #endif
                //atomicAdd(dEtot, En - E0);
            }
        }
    }

    //blockDeltaE[t_id] = localDeltaE;
    // need to synchronize again to avoid data runs
    #ifdef CHECKSUM
    __syncthreads();  
    if(t_id==0) {atomicAdd(dEtot, deltaE);}
    #endif
    
    // save rng state
    states[gidx] = localState;
    }
}
__global__ void Etot(int* dz, int L, double J, double B, double kbT,
                           curandState* states, double* Etot){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=L*L) return;


    
    double E0=0;
    int neighbours[4]={0,0,0,0};
    if(i>L*L-L-1) neighbours[0]=i%L; else neighbours[0]=i+L;
    if(i<L) neighbours[1]=L*L-L+i; else neighbours[1]=i-L;
    if(i%L==0) neighbours[2]=i+L-1; else neighbours[2]=i-1;
    if((i+1)%L==0) neighbours[3]=i-L+1; else neighbours[3]=i+1;
    for (int k : {0, 3}) {
            int j = neighbours[k];
            int Sjz = dz[j];

            E0 -= J * dz[i]*Sjz;

            //double cross_x = spin[1]*Sjz - spin[2]*Sjy;
            //double cross_y = spin[2]*Sjx - spin[0]*Sjz;
            //E -= atoms.D * (dx[k]*cross_x + dy[k]*cross_y);
        }
    E0-=B*dz[i];
    atomicAdd(Etot,E0);
    
}

void PrintSys(System& sys, ofstream& file){
    Atoms& atoms=sys.atoms;
    for(size_t i=0;i<atoms.no_atoms;i++){
        size_t x= i%sys.size;
        size_t y= i/sys.size;
        file<<endl<<x<<" "<<y<<" "<< atoms.z[i];
        if(x==(sys.size-1)){
            file<<endl<<x+1<<" "<<y<<" "<< atoms.z[i-sys.size+1];
        }
        if(y==(sys.size-1)){
            file<<endl<<x<<" "<<y+1<<" "<< atoms.z[i%sys.size];
        }
    }
    file<<endl<<sys.size<<" "<<sys.size<<" "<< atoms.z[0];
}

double Mag(System& sys){
    Atoms& atoms=sys.atoms;
    double M=0;
    for(size_t i=0;i<atoms.no_atoms;i++){
        M+=atoms.z[i];
    }

    M/=atoms.no_atoms;
    return M;
}
__global__ void kern_Mag(int* dz, int L, double* M){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<L*L) atomicAdd(M,(double)dz[i]/(double)(L*L));
}

void MonteCarlo(System& sys, ofstream& filenameE, ofstream& filename, size_t steps,size_t fwrite){
    double Ei=0;
    double Ef=0;
    double M=0;
    Atoms& atoms=sys.atoms;
    auto tstart = std::chrono::high_resolution_clock::now();
    // Compute initial energy on host (same as original)


    // initialize the GPU parallelization 
    int L=(int)sys.size;
    int noAtoms=L*L;
    const int blockSize=tile_w*tile_h;
    //const int gridSize=(noAtoms+blockSize-1)/blockSize;

    // 2d grid dimensions
    int num_tiles_x = (L + tile_w -1)/tile_w;
    int num_tiles_y = (L + tile_h -1) / tile_h; 

    dim3 gridDim(num_tiles_x, num_tiles_y); // number of blocks essentially 
    dim3 blockDim(blockSize); 
    
    int* dz; curandState* dStates; double* dEtot; double* dEi; double* dEf; double* dM;
    
    cudaCheck(cudaMalloc(&dz,       noAtoms*sizeof(int)),"alloc dz");
    cudaCheck(cudaMalloc(&dStates,  noAtoms*sizeof(curandState)), "dStates");
    cudaCheck(cudaMalloc(&dEtot,    sizeof(double)),"dEtot");
    cudaCheck(cudaMalloc(&dEi,    sizeof(double)),"dEi alloc");
    cudaCheck(cudaMalloc(&dEf,    sizeof(double)),"dEf alloc");
    cudaCheck(cudaMalloc(&dM,    sizeof(double)),"dM alloc");


    cudaCheck(cudaMemcpy(dz, atoms.z.data(), noAtoms*sizeof(int), cudaMemcpyHostToDevice), "dz copy");
    cudaCheck(cudaMemset(dEtot, 0, sizeof(double)), "dEtot set");
    cudaCheck(cudaMemset(dEi, 0, sizeof(double)),"dEi set");
    cudaCheck(cudaMemset(dEf, 0, sizeof(double)),"dEf set");
    cudaCheck(cudaMemset(dM, 0, sizeof(double)),"dM set");

    unsigned long long seed=(unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // RNG gets initialized per block so it has to be number of blocks, number of threads

    int numThreads = tile_w*tile_h; 
    int numBlocks = (noAtoms + numThreads -1)/ numThreads; // works by example, e.g. if 32x32, numblocks = 1024 +256 -1 / 256 = 4 which makes sense 
    
    InitRng<<<numBlocks, numThreads>>>(dStates, seed, noAtoms); // initializes numblocks x numthreads which is the total number of atoms, 1 per atom
                                                                // could have kept the variable names the same as before but it"s easier to follow that way

    
    cudaCheck(cudaGetLastError(),"init rng"); 
    cudaCheck(cudaDeviceSynchronize(), "first sync");

    Etot<<<numBlocks, numThreads>>>(dz,L,atoms.J,atoms.B,atoms.kbT,dStates,dEi);
    cudaCheck(cudaGetLastError(),"Etot startup"); cudaCheck(cudaDeviceSynchronize(),"sync");
    cudaCheck(cudaMemcpy(&Ei, dEi, sizeof(double), cudaMemcpyDeviceToHost),"copy Ei");
    cudaCheck(cudaFree(dEi), "free dEi");

    // Host-controlled loop: launch kernel per timestep
    for (size_t t = 0; t < steps; t++) {
        // Even parity
        SweepAll<<<gridDim, blockDim>>>(dz, L, atoms.J, atoms.B, atoms.kbT, 
                                                dStates, dEtot, 0);
        cudaCheck(cudaGetLastError(),"sweep even");
        //cudaCheck(cudaDeviceSynchronize());
        
        // Odd parity
        SweepAll<<<gridDim, blockDim>>>(dz, L, atoms.J, atoms.B, atoms.kbT, 
                                                dStates, dEtot, 1);
        cudaCheck(cudaGetLastError(), "sweep odd");
        //cudaCheck(cudaDeviceSynchronize());
        #ifdef CHECKSUM
        if(t%fwrite==0){
        float currentE=0;
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(&currentE, dEtot, sizeof(double), cudaMemcpyDeviceToHost),"write dE copy");
        sys.E=currentE;
        filenameE << t << " " << currentE << endl;
        }
        #endif
    }
     #ifdef CHECKSUM

    double currentE=0;
    cudaCheck(cudaMemcpy(&currentE, dEtot, sizeof(double), cudaMemcpyDeviceToHost),"write dE copy");
    sys.E=currentE;
    filenameE << steps << " " << currentE << endl;
    #endif
    
    // Copy final energy (accumulated across both sweeps)
    

    // Copy final state back to host
    cudaCheck(cudaMemcpy(atoms.z.data(), dz, noAtoms*sizeof(int), cudaMemcpyDeviceToHost), "dz copy final");
    Etot<<<numBlocks, numThreads>>>(dz,L,atoms.J,atoms.B,atoms.kbT,dStates,dEf);
    cudaCheck(cudaGetLastError(),"Etot end"); cudaCheck(cudaDeviceSynchronize(),"sync");
    cudaCheck(cudaMemcpy(&Ef, dEf, sizeof(double), cudaMemcpyDeviceToHost),"copy Ef");
    cudaCheck(cudaFree(dEf), "free dEf");


    kern_Mag<<<numBlocks, numThreads>>>(dz,L,dM);
    cudaCheck(cudaGetLastError(),"rng startup"); cudaCheck(cudaDeviceSynchronize(),"synch");
    cudaCheck(cudaMemcpy(&M, dM, sizeof(double), cudaMemcpyDeviceToHost),"copy dM");
    cudaCheck(cudaFree(dM), "free dM");


    
    cudaCheck(cudaFree(dz), "free dz");
    cudaCheck(cudaFree(dStates), "free dStates");
    cudaCheck(cudaFree(dEtot), "free dEtot");

    
    auto tend = std::chrono::high_resolution_clock::now();
    cout<<"Wall time: "<<(tend - tstart).count()*1e-9<<endl<<
        "Initial Energy: "<<Ei<<endl<<
        "Final Energy: "<<Ef<<endl<<
        "Energy Diff: "<<(Ef-Ei)<<endl<<
        #ifdef CHECKSUM
        "Energy change (steps): "<<sys.E<<endl<<
        #endif
        "Magnetization Z: "<<M<<endl;


}

int main(int argc, char* argv[]){
    Sim_Configuration cfg({argv, argv+argc});
    System sys=MakeLattice(cfg.size,cfg.J,cfg.K,cfg.D,cfg.B,cfg.kbT);
    ofstream snaps(cfg.filename);
    ofstream Energy(cfg.filenameE);
    
    //snaps<<"Time=0";
    //PrintSys(sys,snaps);
    
    MonteCarlo(sys,Energy,snaps,cfg.steps,cfg.data_period);
    
    //snaps<<endl<<"Time="<<cfg.steps;
    
    //PrintSys(sys,snaps);


}
