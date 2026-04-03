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
#include <omp.h>
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
    
double Ecalc(Atoms& atoms, int& spin, size_t& i){
    double E=0;
    //const int dx[4] = { 0,  0, -1, 1};
    //const int dy[4] = { 1, -1,  0, 0};
    for (int k = 0; k < 4; k++) {
        int j = atoms.neighbours[i][k];
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
double EcalcTot(Atoms& atoms, int& spin, size_t& i){
    double E=0;
    //const int dx[4] = { 0,  0, -1, 1};
    //const int dy[4] = { 1, -1,  0, 0};
    for (int k : {0, 3}) {
            int j = atoms.neighbours[i][k];
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
double SweepEven(System &sys){
    Atoms& atoms = sys.atoms;
    double Etot=0;
    for(size_t i=0;i<sys.size;i++)
    for(size_t j=0;j<sys.size;j++)
    if((i+j)%2==0){
        size_t k= i+j*sys.size;
        int spin=NewVec();
        int oldspin=atoms.z[k];
        double E0=Ecalc(atoms, oldspin, k);
        double En=Ecalc(atoms, spin, k);
        double p=unifReDistr0to1(mt);

        if(exp(-(En-E0)/(atoms.kbT))>p){
            //cout<<"delta E="<<En-E0<<endl;

            atoms.z[k]=spin;
            Etot+=(En-E0);
        }
        
    }
    return Etot;
    
}

double SweepOdd(System &sys){
    Atoms& atoms = sys.atoms;
    double Etot=0;
    for(size_t i=0;i<sys.size;i++)
    for(size_t j=0;j<sys.size;j++)
    if((i+j+1)%2==0){
        size_t k= i+j*sys.size;
        int spin=NewVec();
        int oldspin=atoms.z[k];
        double E0=Ecalc(atoms, oldspin, k);
        double En=Ecalc(atoms, spin, k);
        double p=unifReDistr0to1(mt);

        if(exp(-(En-E0)/(atoms.kbT))>p){
            //cout<<"delta E="<<En-E0<<endl;

            atoms.z[k]=spin;
            Etot+=(En-E0);
        }
        
    }
    return Etot;
    
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
/*double Qcalc(System& sys){
    Atoms& atoms=sys.atoms;
    double Q=0;
    for(size_t i=0;i<atoms.no_atoms;i++){
        double dx[3]={(atoms.x[atoms.neighbours[i][3]]-atoms.x[atoms.neighbours[i][2]])/2,
                        (atoms.y[atoms.neighbours[i][3]]-atoms.y[atoms.neighbours[i][2]])/2,
                        (atoms.z[atoms.neighbours[i][3]]-atoms.z[atoms.neighbours[i][2]])/2};
        
        double dy[3]={(atoms.x[atoms.neighbours[i][0]]-atoms.x[atoms.neighbours[i][1]])/2,
                        (atoms.y[atoms.neighbours[i][0]]-atoms.y[atoms.neighbours[i][1]])/2,
                        (atoms.z[atoms.neighbours[i][0]]-atoms.z[atoms.neighbours[i][1]])/2};
        Q+=atoms.x[i]*(dx[1]*dy[2]-dx[2]*dy[1])+atoms.y[i]*(dx[2]*dy[0]-dx[0]*dy[2])+atoms.z[i]*(dx[0]*dy[1]-dx[1]*dy[0]);
    }
    Q/=(PI*4.0);
    return Q;
}*/
double Mag(System& sys){
    Atoms& atoms=sys.atoms;
    double M=0;
    for(size_t i=0;i<atoms.no_atoms;i++){
        M+=atoms.z[i];
    }

    M/=atoms.no_atoms;
    return M;
}

void MonteCarlo(System& sys, ofstream& filenameE, ofstream& filename, size_t steps,size_t fwrite){
    double Ei=0;
    double Ef=0;
    Atoms& atoms=sys.atoms;
    auto tstart = std::chrono::high_resolution_clock::now();
    for(size_t i=0;i<atoms.no_atoms;i+=1){
        int oldspin=atoms.z[i];
        Ei+=EcalcTot(sys.atoms,oldspin,i);
    }
    
    
    
    for(size_t t=0;t<steps;t++){
        double E=0;
        
        E=SweepEven(sys);
        
        E+=SweepOdd(sys);
        
        
        sys.E+=E;
        if(t%fwrite==0){
            filenameE<<t<<" "<<sys.E<<endl;
        }
    }
    for(size_t i=0;i<atoms.no_atoms;i+=1){
        int oldspin=atoms.z[i];
        Ef+=EcalcTot(sys.atoms,oldspin,i);
    }
    
    //double Q=Qcalc(sys);
    double M=Mag(sys);
    auto tend = std::chrono::high_resolution_clock::now();
    cout<<"Wall time: "<<(tend - tstart).count()*1e-9<<endl<<
        "Initial Eneergy: "<<Ei<<endl<<
        "Final Energy: "<<Ef<<endl<<
        "Energy Diff: "<<(Ef-Ei)<<endl<<
        "Energy change (steps): "<<sys.E<<endl<<
        "Magnetization Z: "<<M<<endl;


}

int main(int argc, char* argv[]){
    Sim_Configuration cfg({argv, argv+argc});
    System sys=MakeLattice(cfg.size,cfg.J,cfg.K,cfg.D,cfg.B,cfg.kbT);
    ofstream snaps(cfg.filename);
    ofstream Energy(cfg.filenameE);
    
    snaps<<"Time=0";
    PrintSys(sys,snaps);
    
    MonteCarlo(sys,Energy,snaps,cfg.steps,cfg.data_period);
    
    snaps<<endl<<"Time="<<cfg.steps;
    
    PrintSys(sys,snaps);


}

    