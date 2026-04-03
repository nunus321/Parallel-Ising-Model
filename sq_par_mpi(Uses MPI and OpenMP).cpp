#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <cassert>

#include <mpi.h>
#include <omp.h>

using namespace std;

// each thread gets its own rng
static int mpi_rank_for_seed = 0;

thread_local mt19937 rng_global(std::random_device{}());

// reseed with rank + thread id so every thread has a unique stream
inline void reseed_rng() {
    std::random_device rd_local;
    rng_global.seed(rd_local()
         ^ (static_cast<unsigned>(mpi_rank_for_seed) * 2654435761u)
         ^ (static_cast<unsigned>(omp_get_thread_num()) * 2246822519u));
}

int NewVec() {
    std::uniform_int_distribution<int> d(0, 1);
    return d(rng_global) * 2 - 1;          // +1 or -1
}

double RandDouble() {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(rng_global);
}

// command-line args
class Sim_Configuration {
public:
    size_t steps       = 10000;
    size_t size        = 100;      // global lattice side length L
    size_t data_period = 100;
    double J   = 1;
    double K   = 0;
    double kbT = 0.001;
    double D   = 0;
    double B   = 1;
    string init_config = "";
    string filename  = "Config.dat";
    string filenameE = "Energy.dat";

    Sim_Configuration(vector<string> argument) {
        for (size_t i = 1; i < argument.size(); i += 2) {
            string arg = argument.at(i);
            if      (arg == "-h") {
                cout << "MC -steps <number of steps> -size <Lattice side>"
                     << " -J <Coupling energy> -D <DMI strength>"
                     << " -B <Magnetic Field> -K <Anisotropy strength> -kbT <Temperature>"
                     << " -init <initial config type> -fwrite <io frequency>"
                     << " -ofileE <filename Energies> -ofile <filename Snapshots>\n";
                MPI_Finalize();
                exit(0);
            }
            else if (arg == "-steps")  steps       = stoi(argument[i+1]);
            else if (arg == "-size")   size        = stoi(argument[i+1]);
            else if (arg == "-fwrite") data_period = stoi(argument[i+1]);
            else if (arg == "-ofile")  filename    = argument[i+1];
            else if (arg == "-ofileE") filenameE   = argument[i+1];
            else if (arg == "-J")      J   = stod(argument[i+1]);
            else if (arg == "-D")      D   = stod(argument[i+1]);
            else if (arg == "-B")      B   = stod(argument[i+1]);
            else if (arg == "-K")      K   = stod(argument[i+1]);
            else if (arg == "-kbT")    kbT = stod(argument[i+1]);
            else if (arg == "-init")   init_config = argument[i+1];
            else {
                cout << "---> error: the argument type " << arg
                     << " is not recognized\n";
            }
        }
    }
};

// each rank owns some rows of the lattice plus 2 ghost rows (one above, one below)
class LocalLattice {
public:
    int    L;
    int    local_rows;
    int    global_row0;
    vector<int> z;

    double J, K, D, B, kbT;

    int rank_up, rank_down;

    LocalLattice() : L(0), local_rows(0), global_row0(0),
                     J(0), K(0), D(0), B(0), kbT(0),
                     rank_up(0), rank_down(0) {}

    int total_rows() const { return local_rows + 2; }

    inline size_t idx(int lr, int col) const {
        return static_cast<size_t>(lr) * L + col;
    }

    // fill owned rows with random spins
    void rand_init() {
        mt19937 rng_init(random_device{}()
                         ^ (static_cast<unsigned>(global_row0) * 2654435761u));
        uniform_int_distribution<int> d(0, 1);
        for (int r = 1; r <= local_rows; ++r)
            for (int c = 0; c < L; ++c)
                z[idx(r, c)] = d(rng_init) * 2 - 1;
    }
};

// split the L rows across ranks
LocalLattice MakeLocalLattice(int L, int nprocs, int rank,
                              double J, double K, double D,
                              double B, double kbT)
{
    LocalLattice lat;
    lat.L   = L;
    lat.J   = J;  lat.K = K;  lat.D = D;  lat.B = B;  lat.kbT = kbT;

    int base      = L / nprocs;
    int remainder = L % nprocs;
    if (rank < remainder) {
        lat.local_rows = base + 1;
        lat.global_row0 = rank * (base + 1);
    } else {
        lat.local_rows = base;
        lat.global_row0 = remainder * (base + 1) + (rank - remainder) * base;
    }

    lat.rank_up   = (rank - 1 + nprocs) % nprocs;
    lat.rank_down = (rank + 1) % nprocs;

    lat.z.assign(static_cast<size_t>(lat.total_rows()) * L, 0);
    lat.rand_init();

    return lat;
}

// swap ghost rows with neighbours
void exchange_ghosts(LocalLattice& lat) {
    int L = lat.L;
    MPI_Status status;

    MPI_Sendrecv(
        &lat.z[lat.idx(1, 0)],                L, MPI_INT, lat.rank_up,   0,
        &lat.z[lat.idx(lat.local_rows+1, 0)], L, MPI_INT, lat.rank_down, 0,
        MPI_COMM_WORLD, &status);

    MPI_Sendrecv(
        &lat.z[lat.idx(lat.local_rows, 0)],   L, MPI_INT, lat.rank_down, 1,
        &lat.z[lat.idx(0, 0)],                L, MPI_INT, lat.rank_up,   1,
        MPI_COMM_WORLD, &status);
}

// energy of one spin using all 4 neighbours
double Ecalc(const LocalLattice& lat, int spin, int lr, int col) {
    double E = 0.0;
    int L = lat.L;

    int up_r   = lr - 1;
    int down_r = lr + 1;
    int left_c = (col == 0)   ? L - 1 : col - 1;
    int right_c= (col == L-1) ? 0     : col + 1;

    E -= lat.J * spin * lat.z[lat.idx(up_r,   col)];
    E -= lat.J * spin * lat.z[lat.idx(down_r,  col)];
    E -= lat.J * spin * lat.z[lat.idx(lr,      left_c)];
    E -= lat.J * spin * lat.z[lat.idx(lr,      right_c)];

    E -= lat.B * spin;
    return E;
}

// energy using only down+right neighbours (so each bond is counted once)
double EcalcTot(const LocalLattice& lat, int spin, int lr, int col) {
    double E = 0.0;
    int L = lat.L;

    int down_r  = lr + 1;
    int right_c = (col == L-1) ? 0 : col + 1;

    E -= lat.J * spin * lat.z[lat.idx(down_r, col)];
    E -= lat.J * spin * lat.z[lat.idx(lr,     right_c)];

    E -= lat.B * spin;
    return E;
}

// update even sites
double SweepEven(LocalLattice& lat) {
    double Etot = 0.0;
    int lr_max = lat.local_rows;
    int L      = lat.L;
    int grow0  = lat.global_row0;

    #pragma omp parallel for collapse(2) schedule(static) reduction(+:Etot)
    for (int lr = 1; lr <= lr_max; ++lr) {
        for (int col = 0; col < L; ++col) {
            int global_row = grow0 + (lr - 1);
            if ((global_row + col) % 2 == 0) {
                int spin    = NewVec();
                int oldspin = lat.z[lat.idx(lr, col)];
                double E0   = Ecalc(lat, oldspin, lr, col);
                double En   = Ecalc(lat, spin,    lr, col);
                double p    = RandDouble();
                if (exp(-(En - E0) / lat.kbT) > p) {
                    lat.z[lat.idx(lr, col)] = spin;
                    Etot += (En - E0);
                }
            }
        }
    }
    return Etot;
}

// update odd sites
double SweepOdd(LocalLattice& lat) {
    double Etot = 0.0;
    int lr_max = lat.local_rows;
    int L      = lat.L;
    int grow0  = lat.global_row0;

    #pragma omp parallel for collapse(2) schedule(static) reduction(+:Etot)
    for (int lr = 1; lr <= lr_max; ++lr) {
        for (int col = 0; col < L; ++col) {
            int global_row = grow0 + (lr - 1);
            if ((global_row + col) % 2 != 0) {
                int spin    = NewVec();
                int oldspin = lat.z[lat.idx(lr, col)];
                double E0   = Ecalc(lat, oldspin, lr, col);
                double En   = Ecalc(lat, spin,    lr, col);
                double p    = RandDouble();
                if (exp(-(En - E0) / lat.kbT) > p) {
                    lat.z[lat.idx(lr, col)] = spin;
                    Etot += (En - E0);
                }
            }
        }
    }
    return Etot;
}

// total energy across all ranks
double TotalEnergy(LocalLattice& lat) {
    exchange_ghosts(lat);

    double localE = 0.0;
    int lr_max = lat.local_rows;
    int L      = lat.L;

    #pragma omp parallel for collapse(2) schedule(static) reduction(+:localE)
    for (int lr = 1; lr <= lr_max; ++lr)
        for (int col = 0; col < L; ++col) {
            int spin = lat.z[lat.idx(lr, col)];
            localE += EcalcTot(lat, spin, lr, col);
        }

    double globalE = 0.0;
    MPI_Allreduce(&localE, &globalE, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalE;
}

// magnetisation per spin
double Mag(LocalLattice& lat, int global_N) {
    double localM = 0.0;
    int lr_max = lat.local_rows;
    int L      = lat.L;

    #pragma omp parallel for collapse(2) schedule(static) reduction(+:localM)
    for (int lr = 1; lr <= lr_max; ++lr)
        for (int col = 0; col < L; ++col)
            localM += lat.z[lat.idx(lr, col)];

    double globalM = 0.0;
    MPI_Allreduce(&localM, &globalM, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalM / global_N;
}

// collect the full lattice on rank 0 for output
vector<int> GatherLattice(LocalLattice& lat, int nprocs, int rank) {
    int L = lat.L;

    // pack owned rows
    vector<int> local_owned(lat.local_rows * L);
    for (int r = 1; r <= lat.local_rows; ++r)
        for (int c = 0; c < L; ++c)
            local_owned[(r - 1) * L + c] = lat.z[lat.idx(r, c)];

    // gatherv because ranks can have different row counts
    vector<int> recvcounts, displs;
    vector<int> global;

    if (rank == 0) {
        recvcounts.resize(nprocs);
        displs.resize(nprocs);
    }

    int send_count = lat.local_rows * L;
    MPI_Gather(&send_count, 1, MPI_INT,
               (rank == 0 ? recvcounts.data() : nullptr), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i)
            displs[i] = displs[i-1] + recvcounts[i-1];
        global.resize(static_cast<size_t>(L) * L);
    }

    MPI_Gatherv(local_owned.data(), send_count, MPI_INT,
                (rank == 0 ? global.data() : nullptr),
                (rank == 0 ? recvcounts.data() : nullptr),
                (rank == 0 ? displs.data() : nullptr),
                MPI_INT, 0, MPI_COMM_WORLD);

    return global;
}

// write lattice to file
void PrintSys(const vector<int>& global, int L, ofstream& file) {
    for (int i = 0; i < L * L; ++i) {
        int x = i % L;
        int y = i / L;
        file << "\n" << x << " " << y << " " << global[i];
        if (x == L - 1)
            file << "\n" << x + 1 << " " << y << " " << global[i - L + 1];
        if (y == L - 1)
            file << "\n" << x << " " << y + 1 << " " << global[i % L];
    }
    file << "\n" << L << " " << L << " " << global[0];
}

// mc loop
void MonteCarlo(LocalLattice& lat, int nprocs, int rank,
                ofstream& fileE, ofstream& fileSnap,
                size_t steps, size_t fwrite, int L)
{
    auto tstart = chrono::high_resolution_clock::now();

    for (size_t t = 0; t < steps; ++t) {
        exchange_ghosts(lat);
        SweepEven(lat);
        exchange_ghosts(lat);
        SweepOdd(lat);
    }

    auto tend = chrono::high_resolution_clock::now();

    if (rank == 0) {
        cout << "Wall time: "
             << (tend - tstart).count() * 1e-9 << endl;
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    mpi_rank_for_seed = rank;
    reseed_rng();
    #pragma omp parallel
    { reseed_rng(); }

    Sim_Configuration cfg({argv, argv + argc});
    int L = static_cast<int>(cfg.size);

    if (L < nprocs) {
        if (rank == 0)
            cerr << "Error: lattice size (" << L
                 << ") must be >= number of MPI ranks (" << nprocs << ")\n";
        MPI_Finalize();
        return 1;
    }

    LocalLattice lat = MakeLocalLattice(L, nprocs, rank,
                                        cfg.J, cfg.K, cfg.D,
                                        cfg.B, cfg.kbT);

    ofstream snaps, energy;

    MonteCarlo(lat, nprocs, rank, energy, snaps,
               cfg.steps, cfg.data_period, L);

    MPI_Finalize();
    return 0;
}
