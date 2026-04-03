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
#include <curand_kernel.h>  // replaces <omp.h>; cuRAND provides on-device RNG
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

const double PI=3.141592653287979;

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
    if(tid<n) curand_init(seed,(unsigned long long)tid,0,&states[tid]);
}

// Parallel sweep over even-parity sites: (x+y)%2==0
// Parity is based on 2D coordinates — required for correct parallel independence
__global__ void SweepEven(int* dz, int L, double J, double B, double kbT,
                           curandState* states, double* Etot){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=L*L) return;
    int x=i%L, y=i/L;
    if((x+y)%2!=0) return;

    curandState localState=states[i];
    int spin=(curand_uniform(&localState)<0.5f)?-1:1;
    int oldspin=dz[i];
    double E0=Ecalc(dz,oldspin,i,L,J,B);
    double En=Ecalc(dz,spin,i,L,J,B);
    double p=curand_uniform_double(&localState);

    if(exp(-(En-E0)/kbT)>p){
        //printf("delta E=%f\n",En-E0);

        dz[i]=spin;
        #ifdef CHECKSUM
        atomicAdd(Etot,(En-E0));
        #endif
    }
    states[i]=localState;
}

// Parallel sweep over odd-parity sites: (x+y)%2==1
__global__ void SweepOdd(int* dz, int L, double J, double B, double kbT,
                          curandState* states, double* Etot){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=L*L) return;
    int x=i%L, y=i/L;
    if((x+y)%2!=1) return;

    curandState localState=states[i];
    int spin=(curand_uniform(&localState)<0.5f)?-1:1;
    int oldspin=dz[i];
    double E0=Ecalc(dz,oldspin,i,L,J,B);
    double En=Ecalc(dz,spin,i,L,J,B);
    double p=curand_uniform_double(&localState);

    if(exp(-(En-E0)/kbT)>p){
        //printf("delta E=%f\n",En-E0);

        dz[i]=spin;
        #ifdef CHECKSUM
        atomicAdd(Etot,(En-E0));
        #endif
    }
    states[i]=localState;
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
    if(i>=L*L)return;
    atomicAdd(M,(double)dz[i]/(double)(L*L));
}

void MonteCarlo(System& sys, ofstream& filenameE, ofstream& filename, size_t steps,size_t fwrite){
    double Ei=0;
    double Ef=0;
    Atoms& atoms=sys.atoms;
    auto tstart = std::chrono::high_resolution_clock::now();
    // Compute initial energy on host (same as original)
    

    // === GPU setup ===
    int L=(int)sys.size;
    int noAtoms=L*L;
    const int blockSize=256;
    const int gridSize=(noAtoms+blockSize-1)/blockSize;

    int* dz; curandState* dStates; double* dEtot; double* dEi; double* dEf; double* dM;
    cudaCheck(cudaMalloc(&dz,       noAtoms*sizeof(int)),"dz alloc");
    cudaCheck(cudaMalloc(&dStates,  noAtoms*sizeof(curandState)),"dStates alloc");
    cudaCheck(cudaMalloc(&dEtot,    sizeof(double)),"dEtot alloc");
    cudaCheck(cudaMalloc(&dEi,    sizeof(double)),"dEi alloc");
    cudaCheck(cudaMalloc(&dEf,    sizeof(double)),"dEf alloc");
    cudaCheck(cudaMalloc(&dM,    sizeof(double)),"dM alloc");

    cudaCheck(cudaMemcpy(dz, atoms.z.data(), noAtoms*sizeof(int), cudaMemcpyHostToDevice),"dz copy");
    cudaCheck(cudaMemset(dEtot, 0, sizeof(double)),"dEtot set");
    cudaCheck(cudaMemset(dEi, 0, sizeof(double)),"dEi set");
    cudaCheck(cudaMemset(dEf, 0, sizeof(double)),"dEf set");
    cudaCheck(cudaMemset(dM, 0, sizeof(double)),"dM set");
    
    unsigned long long seed=(unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    InitRng<<<gridSize,blockSize>>>(dStates, seed, noAtoms);
    cudaCheck(cudaGetLastError(),"rng startup"); cudaCheck(cudaDeviceSynchronize(),"sync");
    //int oldspin=atoms.z[i];
    //Ei+=EcalcTot(sys,oldspin,i);
    Etot<<<gridSize,blockSize>>>(dz,L,atoms.J,atoms.B,atoms.kbT,dStates,dEi);
    cudaCheck(cudaGetLastError(),"Etot startup"); cudaCheck(cudaDeviceSynchronize(),"sync");
    cudaCheck(cudaMemcpy(&Ei, dEi, sizeof(double), cudaMemcpyDeviceToHost),"copy Ei");
    cudaCheck(cudaFree(dEi), "free dEi");

    
    
    for(size_t t=0;t<steps;t++){
        
        SweepEven<<<gridSize,blockSize>>>(dz, L, atoms.J, atoms.B, atoms.kbT, dStates, dEtot);
        cudaCheck(cudaGetLastError(),"sweep even");
        
        SweepOdd<<<gridSize,blockSize>>>(dz, L, atoms.J, atoms.B, atoms.kbT, dStates, dEtot);
        cudaCheck(cudaGetLastError(), "sweep odd");
        
        #ifdef CHECKSUM
        if(t%fwrite==0){
            cudaCheck(cudaDeviceSynchronize(),"sync to write");
            cudaCheck(cudaMemcpy(&sys.E, dEtot, sizeof(double), cudaMemcpyDeviceToHost),"copy to write");
            filenameE<<t<<" "<<sys.E<<endl;
        }
        #endif
    }
    cudaCheck(cudaDeviceSynchronize(),"final sync");

    // Copy final spins back to host
    cudaCheck(cudaMemcpy(atoms.z.data(), dz, noAtoms*sizeof(int), cudaMemcpyDeviceToHost), "final copy dz");
    #ifdef CHECKSUM
    cudaCheck(cudaMemcpy(&sys.E, dEtot, sizeof(double), cudaMemcpyDeviceToHost), "final copy E");
    #endif

    

    // Compute final energy on host using updated atoms.z (same as original)
    Etot<<<gridSize,blockSize>>>(dz,L,atoms.J,atoms.B,atoms.kbT,dStates,dEf);
    cudaCheck(cudaGetLastError(),"Etot end"); cudaCheck(cudaDeviceSynchronize(),"sync");
    cudaCheck(cudaMemcpy(&Ef, dEf, sizeof(double), cudaMemcpyDeviceToHost),"copy Ef");
     cudaCheck(cudaFree(dEf), "free dEf");
    //double Q=Qcalc(sys);
    double M=0;
    kern_Mag<<<gridSize,blockSize>>>(dz,L,dM);
    cudaCheck(cudaGetLastError(),"rng startup"); cudaCheck(cudaDeviceSynchronize(),"synch");
    cudaCheck(cudaMemcpy(&M, dM, sizeof(double), cudaMemcpyDeviceToHost),"copy dM");
     cudaCheck(cudaFree(dM), "free dM");
    cudaCheck(cudaFree(dz), "free dz"); cudaCheck(cudaFree(dStates), "free rng states"); cudaCheck(cudaFree(dEtot), "free dEtot");
        
       
    
    auto tend = std::chrono::high_resolution_clock::now();
    cout<<"Wall time: "<<(tend - tstart).count()*1e-9<<endl<<
        "Initial Eneergy: "<<Ei<<endl<<
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
