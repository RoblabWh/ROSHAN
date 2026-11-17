// Standalone transcription of the old ROSHAN FireSPIN loop on its uniform map (generic class),
// including its quirks: int-truncated convection spawn, cyclic buffer of normals, mt19937.
// Sources: firemodel_gridmap.cpp (UpdateParticles, UpdateCells), firemodel_firecell.cpp,
// particles/virtual_particle.cpp, particles/radiation_particle.cpp, used_config.yaml.
// usage: old <wind_ms> <seed> <end_s> <block>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>
const int N = 50; const double CELL = 10.0, DT = 0.1;
const double TAU_V = 10, YST_V = 1, YLIM_V = 0.2, FL = 0.15, C0 = 1.98, LT = 80, A = 0.314;
const double YST_R = 1, YLIM_R = 0.165, LR0 = 5, LR1 = 50, SF_MEAN = 0.1, SF_STD = 0.025;
const int NCONV = 10, NRAD = 4;
std::mt19937 gen;
std::vector<double> buf; size_t bi = 0;
double nextN() { if (bi >= buf.size()) bi = 0; return buf[bi++]; }
struct VP { double X[2], U[2], Y; };
struct RP { double X[2], Xmc[2], r, phi, Y, tau; };
struct Cell { int state = 0; double burn, tau_ign, tick = 0, btick = 0; int last = 0, thrc, thrr; };
int main(int argc, char** argv) {
    double uw = atof(argv[1]); int seed = atoi(argv[2]); double end = atof(argv[3]); int block = atoi(argv[4]);
    gen.seed(seed);
    buf.resize(N * N * 30); { std::normal_distribution<double> d(0, 1); for (auto& v : buf) v = d(gen); }
    std::vector<Cell> cells(N * N);
    for (int x = 0; x < N; x++) for (int y = 0; y < N; y++) {
        Cell& c = cells[x * N + y]; c.burn = 120; c.tau_ign = 100;
        std::seed_seq seq{seed, x * 10, y * 10}; std::mt19937 g(seq);
        std::uniform_real_distribution<> dis(0.1, 0.2); std::uniform_int_distribution<> sg(-1, 1);
        c.burn += sg(g) * c.burn * dis(g); c.tau_ign += sg(g) * c.tau_ign * dis(g);
        c.thrc = (c.burn - 1) / NCONV; if (c.thrc < 1) c.thrc = 1;
        c.thrr = (c.burn - 1) / NRAD; if (c.thrr < 1) c.thrr = 1;
    }
    std::set<int> ticking, burning; std::vector<VP> vps; std::vector<RP> rps;
    std::vector<unsigned> visited(N * N, 0); unsigned epoch = 0; int ignited = 0;
    auto ignite = [&](int id) { cells[id].state = 1; cells[id].tick = cells[id].tau_ign; burning.insert(id); ticking.erase(id); ignited++; };
    for (int x = 25 - block / 2; x < 25 - block / 2 + block; x++) for (int y = 25 - block / 2; y < 25 - block / 2 + block; y++) ignite(x * N + y);
    std::uniform_real_distribution<> u01(0, 1);
    double uwi[2] = {uw, 0}, up = A * uw;
    printf("t_s,burning,ignited,particles\n");
    long steps = lround(end / DT);
    for (long s = 1; s <= steps; s++) {
        ++epoch;
        auto visit = [&](double x, double y, bool capable) -> bool {
            int i = floor(x / CELL), j = floor(y / CELL);
            if (i < 0 || j < 0 || i >= N || j >= N || !capable) return false;
            visited[i * N + j] = epoch; if (cells[i * N + j].state == 0) ticking.insert(i * N + j); return true; };
        for (size_t p = 0; p < vps.size();) { VP& v = vps[p];
            v.Y -= v.Y / TAU_V * DT;
            double f1 = ((2.0 + 3.0 * C0) / 4.0) * (up / LT), f2 = pow(C0 * pow(up, 3) / LT * DT, 0.5);
            for (int i = 0; i < 2; i++) { double n = nextN(); v.U[i] += -(f1 * (v.U[i] - uwi[i]) * DT) + f2 * n; v.X[i] += FL * v.U[i] * DT; }
            if (!visit(v.X[0], v.X[1], v.Y >= YLIM_V)) { vps[p] = vps.back(); vps.pop_back(); } else p++; }
        for (size_t p = 0; p < rps.size();) { RP& r = rps[p];
            double nphi = nextN(), nr = nextN();
            r.phi += 2 * M_PI * nphi; r.r += SF_MEAN * DT + SF_STD * sqrt(DT) * nr;
            r.X[0] = r.Xmc[0] + r.r * cos(r.phi); r.X[1] = r.Xmc[1] + r.r * sin(r.phi); r.Y -= r.Y / r.tau * DT;
            if (!visit(r.X[0], r.X[1], r.Y >= YLIM_R)) { rps[p] = rps.back(); rps.pop_back(); } else p++; }
        for (auto it = ticking.begin(); it != ticking.end();) if (visited[*it] != epoch) it = ticking.erase(it); else ++it;
        for (auto it = ticking.begin(); it != ticking.end();) { Cell& c = cells[*it]; c.tick += DT;
            if (c.tick >= c.tau_ign) { int id = *it; it = ticking.erase(it); ignite(id); } else ++it; }
        for (auto it = burning.begin(); it != burning.end();) { Cell& c = cells[*it]; int x = *it / N, y = *it % N;
            c.last = c.btick; c.burn -= DT; c.btick += DT; if (c.burn <= 0) c.state = 2;
            int bt = (int)c.btick; bool is_new = ceil(c.last) != floor(c.btick);
            if (is_new && bt % c.thrc == 0) { double a = u01(gen), b = u01(gen);
                int px = x * CELL + CELL * a, py = y * CELL + CELL * b; // int truncation as in the old constructor
                vps.push_back({{(double)px, (double)py}, {0, 0}, YST_V}); }
            if (is_new && bt % c.thrr == 0) { double a = u01(gen), b = u01(gen);
                double px = x * CELL + CELL * a, py = y * CELL + CELL * b;
                std::normal_distribution<> nd(SF_MEAN, SF_STD); std::uniform_real_distribution<> ud(LR0, LR1);
                double sf = nd(gen); if (sf < 0) sf = -sf; double lr = ud(gen);
                rps.push_back({{px, py}, {px, py}, 0, 0, YST_R, sf > 0.00925 ? lr / sf : 2703}); }
            if (c.state == 2) it = burning.erase(it); else ++it; }
        if (s % 500 == 0) printf("%g,%zu,%d,%zu\n", s * DT, burning.size(), ignited, vps.size() + rps.size());
    }
}
