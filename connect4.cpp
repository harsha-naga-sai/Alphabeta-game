// Connect 4 AI engine (C++17): bitboards + negamax with alpha-beta pruning.
//
// Build:  g++ -O2 -std=c++17 connect4.cpp -o c4
//
// Usage:
//   ./c4 bench [maxDepth]             node counts and speed from the empty board (CSV)
//   ./c4 play <depthA> <depthB> <N>   AI vs AI tournament, one CSV row per game
//   ./c4 best <depth> <moves>         best move after a move string, e.g. "3344"
//   ./c4 compare [maxDepth]           nodes searched WITH vs WITHOUT alpha-beta pruning (CSV)
//
// Board layout: bit index = column * 7 + row (row 0 is the bottom).
// Row 6 of each column is an empty "sentinel" bit so shifts never wrap between columns.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace std;
typedef uint64_t u64;
typedef chrono::steady_clock Clock;

static const int COLS = 7, ROWS = 6;
static const int WIN = 100000, INF = 1000000000;
static const int ORDER[COLS] = {3, 2, 4, 1, 5, 0, 6};   // centre columns first = better pruning
static const int SC_ME[5]  = {0, 0, 3, 12, 0};          // my pieces in an open window of 4
static const int SC_OPP[5] = {0, 0, 3, 14, 0};          // opponent pieces in an open window
static const u64 CENTER = 0x3FULL << 21;                 // the 6 cells of column 3

static vector<u64> WINDOWS;    // all 69 possible four-in-a-row lines as bitmasks
static long long nodes = 0;    // positions searched (for benchmarking)
static bool PRUNE = true;      // false = plain minimax (used only to measure what pruning saves)

static void initWindows() {
    const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    for (int c = 0; c < COLS; c++)
        for (int r = 0; r < ROWS; r++)
            for (auto& d : dirs) {
                int ec = c + 3 * d[0], er = r + 3 * d[1];
                if (ec < 0 || ec >= COLS || er < 0 || er >= ROWS) continue;
                u64 mask = 0;
                for (int k = 0; k < 4; k++) mask |= 1ULL << ((c + k * d[0]) * 7 + (r + k * d[1]));
                WINDOWS.push_back(mask);
            }
}

struct Pos {
    u64 b[2];       // b[0] = first player's stones, b[1] = second player's
    int h[COLS];    // bit index of the next free cell in each column
    int moves;
    Pos() : moves(0) {
        b[0] = b[1] = 0;
        for (int c = 0; c < COLS; c++) h[c] = c * 7;
    }
    bool can(int c) const { return h[c] < c * 7 + ROWS; }
    void play(int c) { b[moves & 1] |= 1ULL << h[c]; h[c]++; moves++; }
    void undo(int c) { moves--; h[c]--; b[moves & 1] ^= 1ULL << h[c]; }
};

// Four in a row? Check vertical, horizontal and both diagonals with shifts.
static inline bool isWin(u64 bb) {
    static const int dirs[4] = {1, 7, 6, 8};
    for (int d : dirs) {
        u64 t = bb & (bb >> d);
        if (t & (t >> (2 * d))) return true;
    }
    return false;
}

// Heuristic score from the point of view of the player to move.
static int evaluate(const Pos& p) {
    int me = p.moves & 1;
    u64 A = p.b[me], B = p.b[me ^ 1];
    int s = 0;
    for (u64 w : WINDOWS) {
        int m = __builtin_popcountll(A & w), o = __builtin_popcountll(B & w);
        if (o == 0) s += SC_ME[m];
        else if (m == 0) s -= SC_OPP[o];
    }
    s += 3 * (__builtin_popcountll(A & CENTER) - __builtin_popcountll(B & CENTER));
    return s;
}

// Negamax with alpha-beta pruning. Returns the score for the player to move.
static int negamax(Pos& p, int depth, int alpha, int beta) {
    nodes++;
    if (p.moves == COLS * ROWS) return 0;                       // draw
    int me = p.moves & 1;
    for (int i = 0; i < COLS; i++) {                            // can I win right now?
        int c = ORDER[i];
        if (p.can(c) && isWin(p.b[me] | (1ULL << p.h[c]))) return WIN - p.moves;
    }
    if (depth == 0) return evaluate(p);
    int best = -INF;
    for (int i = 0; i < COLS; i++) {
        int c = ORDER[i];
        if (!p.can(c)) continue;
        p.play(c);
        int v = -negamax(p, depth - 1, -beta, -alpha);
        p.undo(c);
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        if (PRUNE && alpha >= beta) break;                      // prune
    }
    return best;
}

struct Result { int col; int score; long long nodes; };

static Result bestMove(Pos& p, int depth) {
    nodes = 0;
    int me = p.moves & 1;
    for (int i = 0; i < COLS; i++) {
        int c = ORDER[i];
        if (p.can(c) && isWin(p.b[me] | (1ULL << p.h[c])))
            return {c, WIN - p.moves, 1};
    }
    int alpha = -INF, bestC = -1;
    for (int i = 0; i < COLS; i++) {
        int c = ORDER[i];
        if (!p.can(c)) continue;
        p.play(c);
        int v = -negamax(p, depth - 1, -INF, -alpha);
        p.undo(c);
        if (bestC < 0 || v > alpha) { alpha = v; bestC = c; }
    }
    return {bestC, alpha, nodes};
}

static double msSince(Clock::time_point t0) {
    return chrono::duration<double, milli>(Clock::now() - t0).count();
}

static void runBench(int maxDepth) {
    printf("depth,best_col,score,nodes,ms,nodes_per_sec\n");
    for (int d = 1; d <= maxDepth; d++) {
        Pos p;
        auto t0 = Clock::now();
        Result r = bestMove(p, d);
        double ms = msSince(t0);
        printf("%d,%d,%d,%lld,%.3f,%.0f\n", d, r.col, r.score, r.nodes, ms, ms > 0 ? r.nodes / ms * 1000.0 : 0.0);
        fflush(stdout);
    }
}

static void runCompare(int maxDepth) {
    printf("depth,nodes_pruned,nodes_full,ms_pruned,ms_full,saved_pct\n");
    for (int d = 1; d <= maxDepth; d++) {
        Pos p; PRUNE = true;
        auto t0 = Clock::now(); Result a = bestMove(p, d); double msA = msSince(t0);
        Pos q; PRUNE = false;
        t0 = Clock::now(); Result b = bestMove(q, d); double msB = msSince(t0);
        PRUNE = true;
        printf("%d,%lld,%lld,%.3f,%.3f,%.2f\n", d, a.nodes, b.nodes, msA, msB, 100.0 * (1.0 - (double)a.nodes / (double)b.nodes));
        if (a.col != b.col || a.score != b.score) fprintf(stderr, "MISMATCH at depth %d\n", d);   // pruning must never change the answer
        fflush(stdout);
    }
}

static void runPlay(int dA, int dB, int games) {
    printf("game,a_first,winner,moves,a_moves,b_moves,a_ms,b_ms,a_nodes,b_nodes\n");
    for (int g = 0; g < games; g++) {
        mt19937 rng(1000 + g);
        Pos p;
        bool aFirst = (g % 2 == 0);
        for (int i = 0; i < 2; i++) {                           // 2 random opening moves for variety
            int c;
            do { c = rng() % COLS; } while (!p.can(c));
            p.play(c);
        }
        double ms[2] = {0, 0};
        long long nd[2] = {0, 0};
        int cnt[2] = {0, 0};
        char winner = 'D';
        while (true) {
            int side = p.moves & 1;
            bool isA = (side == 0) == aFirst;
            int who = isA ? 0 : 1;
            auto t0 = Clock::now();
            Result r = bestMove(p, isA ? dA : dB);
            ms[who] += msSince(t0);
            nd[who] += r.nodes;
            cnt[who]++;
            p.play(r.col);
            if (isWin(p.b[side])) { winner = isA ? 'A' : 'B'; break; }
            if (p.moves == COLS * ROWS) break;
        }
        printf("%d,%d,%c,%d,%d,%d,%.3f,%.3f,%lld,%lld\n", g, aFirst ? 1 : 0, winner, p.moves,
               cnt[0], cnt[1], ms[0], ms[1], nd[0], nd[1]);
        fflush(stdout);
    }
}

static void runBest(int depth, const string& seq) {
    Pos p;
    for (char ch : seq) {
        int c = ch - '0';
        if (c < 0 || c >= COLS || !p.can(c)) { fprintf(stderr, "bad move string\n"); exit(1); }
        p.play(c);
    }
    Result r = bestMove(p, depth);
    printf("%d %d %lld\n", r.col, r.score, r.nodes);
}

int main(int argc, char** argv) {
    initWindows();
    if (argc >= 2 && !strcmp(argv[1], "bench")) {
        runBench(argc >= 3 ? atoi(argv[2]) : 8);
    } else if (argc >= 2 && !strcmp(argv[1], "compare")) {
        runCompare(argc >= 3 ? atoi(argv[2]) : 8);
    } else if (argc >= 5 && !strcmp(argv[1], "play")) {
        runPlay(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    } else if (argc >= 4 && !strcmp(argv[1], "best")) {
        runBest(atoi(argv[2]), argv[3]);
    } else {
        printf("Connect 4 AI engine\n"
               "  %s bench [maxDepth]\n"
               "  %s play <depthA> <depthB> <games>\n"
               "  %s best <depth> <moves>\n"
               "  %s compare [maxDepth]\n", argv[0], argv[0], argv[0], argv[0]);
    }
    return 0;
}
