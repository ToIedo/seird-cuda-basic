#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "Error CUDA en " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err) << "\n";            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

enum State : uint8_t {
    SUSCEPTIBLE = 0, // S - Azul oscuro
    EXPOSED     = 1, // E - Amarillo
    INFECTED    = 2, // I - Rojo
    RECOVERED   = 3, // R - Verde
    DEAD        = 4  // D - Gris oscuro
};

struct SimParams {
    float beta  = 0.28f; // Probabilidad de contagio por vecino infectado/día
    int   t_inc = 3;     // Días de incubación (E -> I)
    int   t_inf = 10;    // Días infecciosos (I -> R/D)
    float alpha = 0.04f; // Tasa de letalidad
};

__host__ __device__ inline float rand_uniform(uint32_t& state) {
    uint32_t x = state;
    if (x == 0) x = 0x6D2B79F5u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return static_cast<float>(x) * (1.0f / 4294967296.0f);
}

// ============================================================================
// KERNEL 1: Paso de simulación SEIRD en grilla 2D (1 hilo = 1 celda)
// ============================================================================
__global__ void kernel_seird_step(const uint8_t* __restrict__ state_in,
                                  const uint16_t* __restrict__ timer_in,
                                  uint8_t* __restrict__ state_out,
                                  uint16_t* __restrict__ timer_out,
                                  uint32_t* __restrict__ rng_states,
                                  int N,
                                  SimParams p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    int idx = y * N + x;
    uint8_t  cur_state = state_in[idx];
    uint16_t cur_timer = timer_in[idx];

    uint8_t  next_state = cur_state;
    uint16_t next_timer = cur_timer;

    if (cur_state == SUSCEPTIBLE) {
        int infected_neighbors = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = y + dy;
            if (ny < 0 || ny >= N) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx;
                if (nx < 0 || nx >= N) continue;
                if (state_in[ny * N + nx] == INFECTED) {
                    infected_neighbors++;
                }
            }
        }
        if (infected_neighbors > 0) {
            float p_infect = 1.0f - powf(1.0f - p.beta, static_cast<float>(infected_neighbors));
            uint32_t rng = rng_states[idx];
            if (rand_uniform(rng) < p_infect) {
                next_state = EXPOSED;
                next_timer = 0;
            }
            rng_states[idx] = rng;
        }
    } else if (cur_state == EXPOSED) {
        next_timer = cur_timer + 1;
        if (next_timer >= p.t_inc) {
            next_state = INFECTED;
            next_timer = 0;
        }
    } else if (cur_state == INFECTED) {
        next_timer = cur_timer + 1;
        if (next_timer >= p.t_inf) {
            uint32_t rng = rng_states[idx];
            next_state = (rand_uniform(rng) < p.alpha) ? DEAD : RECOVERED;
            next_timer = 0;
            rng_states[idx] = rng;
        }
    }

    state_out[idx] = next_state;
    timer_out[idx] = next_timer;
}

// ============================================================================
// KERNEL 2: Conteo paralelo de estados (__shared__ + atomicAdd)
// ============================================================================
__global__ void kernel_count_states(const uint8_t* __restrict__ state,
                                    int N,
                                    int* __restrict__ global_counts) {
    __shared__ int local_counts[5];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 5; ++i) local_counts[i] = 0;
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        uint8_t s = state[y * N + x];
        if (s < 5) atomicAdd(&local_counts[s], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < 5; ++i) {
            if (local_counts[i] > 0) atomicAdd(&global_counts[i], local_counts[i]);
        }
    }
}

// ============================================================================
// KERNEL 3: Renderizado directo en GPU a buffer BGRA (0x00RRGGBB para GDI)
// ============================================================================
__global__ void kernel_render_bgra(const uint8_t* __restrict__ state,
                                   uint32_t* __restrict__ pixels,
                                   int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    int idx = y * N + x;
    uint32_t color = 0x00122040; // S: Azul oscuro
    switch (state[idx]) {
        case EXPOSED:   color = 0x00F5C527; break; // E: Amarillo
        case INFECTED:  color = 0x00E62E2D; break; // I: Rojo
        case RECOVERED: color = 0x002ECC71; break; // R: Verde
        case DEAD:      color = 0x003A3A40; break; // D: Gris oscuro
    }
    pixels[idx] = color;
}

// ============================================================================
// VERSIÓN SECUENCIAL EN CPU (para modo --bench)
// ============================================================================
void cpu_seird_step(const std::vector<uint8_t>& state_in,
                    const std::vector<uint16_t>& timer_in,
                    std::vector<uint8_t>& state_out,
                    std::vector<uint16_t>& timer_out,
                    std::vector<uint32_t>& rng_states,
                    int N,
                    SimParams p,
                    int counts[5]) {
    for (int i = 0; i < 5; ++i) counts[i] = 0;
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int idx = y * N + x;
            uint8_t  cur_state = state_in[idx];
            uint16_t cur_timer = timer_in[idx];
            uint8_t  next_state = cur_state;
            uint16_t next_timer = cur_timer;

            if (cur_state == SUSCEPTIBLE) {
                int infected_neighbors = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= N) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        if (nx < 0 || nx >= N) continue;
                        if (state_in[ny * N + nx] == INFECTED) infected_neighbors++;
                    }
                }
                if (infected_neighbors > 0) {
                    float p_infect = 1.0f - std::pow(1.0f - p.beta, static_cast<float>(infected_neighbors));
                    if (rand_uniform(rng_states[idx]) < p_infect) {
                        next_state = EXPOSED;
                        next_timer = 0;
                    }
                }
            } else if (cur_state == EXPOSED) {
                next_timer = cur_timer + 1;
                if (next_timer >= p.t_inc) { next_state = INFECTED; next_timer = 0; }
            } else if (cur_state == INFECTED) {
                next_timer = cur_timer + 1;
                if (next_timer >= p.t_inf) {
                    next_state = (rand_uniform(rng_states[idx]) < p.alpha) ? DEAD : RECOVERED;
                    next_timer = 0;
                }
            }
            state_out[idx] = next_state;
            timer_out[idx] = next_timer;
            counts[next_state]++;
        }
    }
}

void init_population(std::vector<uint8_t>& state,
                     std::vector<uint16_t>& timer,
                     std::vector<uint32_t>& rng,
                     int N,
                     uint32_t run_seed) {
    std::fill(state.begin(), state.end(), SUSCEPTIBLE);
    std::fill(timer.begin(), timer.end(), 0);

    for (size_t i = 0; i < state.size(); ++i) {
        uint32_t s = static_cast<uint32_t>(i * 747796405u + 2891336453u + run_seed);
        rng[i] = (s == 0) ? 1u : s;
    }

    // Sembrar 1 foco central + 20 focos aleatorios para ver ondas de contagio
    auto seed_patch = [&](int cx, int cy) {
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int nx = cx + dx, ny = cy + dy;
                if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                    state[ny * N + nx] = INFECTED;
                }
            }
        }
    };

    seed_patch(N / 2, N / 2);
    uint32_t gen = 123456789u ^ run_seed;
    for (int f = 0; f < 20; ++f) {
        int fx = static_cast<int>(rand_uniform(gen) * (N - 20)) + 10;
        int fy = static_cast<int>(rand_uniform(gen) * (N - 20)) + 10;
        seed_patch(fx, fy);
    }
}

// Estado global simple para la ventana Win32
struct AppState {
    bool running = true;
    bool paused  = false;
    bool reset   = false;
};

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    AppState* app = reinterpret_cast<AppState*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    switch (uMsg) {
        case WM_CLOSE:
        case WM_DESTROY:
            if (app) app->running = false;
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (app) {
                if (wParam == VK_SPACE) app->paused = !app->paused;
                else if (wParam == 'R') app->reset = true;
                else if (wParam == VK_ESCAPE) {
                    app->running = false;
                    PostQuitMessage(0);
                }
            }
            return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int main(int argc, char** argv) {
    int  N     = 512;
    int  days  = 200;
    int  fps   = 30;
    bool bench = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--days") == 0 && i + 1 < argc) days = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc) fps = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--bench") == 0) bench = true;
    }

    const size_t total_cells = static_cast<size_t>(N) * N;
    SimParams params;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "SEIRD-CUDA-Basic | Grilla " << N << "x" << N
              << " | GPU: " << prop.name << "\n";

    std::vector<uint8_t>  h_init_state(total_cells);
    std::vector<uint16_t> h_init_timer(total_cells);
    std::vector<uint32_t> h_init_rng(total_cells);
    uint32_t run_seed = 1u;
    init_population(h_init_state, h_init_timer, h_init_rng, N, run_seed);

    uint8_t  *d_state_a = nullptr, *d_state_b = nullptr;
    uint16_t *d_timer_a = nullptr, *d_timer_b = nullptr;
    uint32_t *d_rng = nullptr, *d_pixels = nullptr;
    int      *d_counts = nullptr;

    CUDA_CHECK(cudaMalloc(&d_state_a, total_cells * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_state_b, total_cells * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_timer_a, total_cells * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_timer_b, total_cells * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_rng,     total_cells * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pixels,  total_cells * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_counts,  5 * sizeof(int)));

    auto upload_initial = [&]() {
        CUDA_CHECK(cudaMemcpy(d_state_a, h_init_state.data(), total_cells * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_timer_a, h_init_timer.data(), total_cells * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rng,     h_init_rng.data(),   total_cells * sizeof(uint32_t), cudaMemcpyHostToDevice));
    };
    upload_initial();

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // ========================================================================
    // MODO BENCHMARK (--bench): Comparación CPU vs GPU sin ventana
    // ========================================================================
    if (bench) {
        float gpu_ms = 0.0f;
        for (int day = 1; day <= days; ++day) {
            CUDA_CHECK(cudaMemset(d_counts, 0, 5 * sizeof(int)));
            CUDA_CHECK(cudaEventRecord(ev_start));
            kernel_seird_step<<<grid, block>>>(d_state_a, d_timer_a, d_state_b, d_timer_b, d_rng, N, params);
            kernel_count_states<<<grid, block>>>(d_state_b, N, d_counts);
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            gpu_ms += ms;
            std::swap(d_state_a, d_state_b);
            std::swap(d_timer_a, d_timer_b);
        }

        std::vector<uint8_t>  c_state_b(total_cells);
        std::vector<uint16_t> c_timer_b(total_cells);
        int cpu_counts[5] = {0};
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int day = 1; day <= days; ++day) {
            cpu_seird_step(h_init_state, h_init_timer, c_state_b, c_timer_b, h_init_rng, N, params, cpu_counts);
            std::swap(h_init_state, c_state_b);
            std::swap(h_init_timer, c_timer_b);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("Benchmark %dx%d (%d dias) | GPU kernels: %.2f ms | CPU: %.2f ms | Speedup: %.2fx\n",
                    N, N, days, gpu_ms, cpu_ms, cpu_ms / gpu_ms);
    } else {
        // ====================================================================
        // MODO VISUAL: Ventana Win32 en tiempo real
        // ====================================================================
        const int WIN_SIZE = 768;
        AppState app;

        WNDCLASSA wc{};
        wc.lpfnWndProc   = WindowProc;
        wc.hInstance     = GetModuleHandleA(nullptr);
        wc.lpszClassName = "SeirdCudaWindow";
        wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
        RegisterClassA(&wc);

        RECT rect{0, 0, WIN_SIZE, WIN_SIZE};
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

        HWND hwnd = CreateWindowExA(
            0, wc.lpszClassName, "SEIRD CUDA - Simulacion en Tiempo Real",
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT, CW_USEDEFAULT,
            rect.right - rect.left, rect.bottom - rect.top,
            nullptr, nullptr, wc.hInstance, nullptr);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&app));

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = N;
        bmi.bmiHeader.biHeight      = -N; // Top-down
        bmi.bmiHeader.biPlanes      = 1;
        bmi.bmiHeader.biBitCount    = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        std::vector<uint32_t> h_pixels(total_cells);
        fs::create_directories("output");
        std::ofstream csv("output/seird_curve.csv");
        csv << "day,S,E,I,R,D\n";

        std::cout << "Controles en ventana: [ESPACIO] Pausar | [R] Reiniciar | [ESC] Salir\n";
        int day = 0;
        HDC hdc = GetDC(hwnd);
        const auto frame_period = std::chrono::milliseconds(1000 / fps);

        while (app.running) {
            auto frame_start = std::chrono::steady_clock::now();

            MSG msg;
            while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessageA(&msg);
            }
            if (!app.running) break;

            if (app.reset) {
                app.reset = false;
                day = 0;
                run_seed += 9973u;
                init_population(h_init_state, h_init_timer, h_init_rng, N, run_seed);
                upload_initial();
            }

            if (!app.paused) {
                day++;
                CUDA_CHECK(cudaMemset(d_counts, 0, 5 * sizeof(int)));
                CUDA_CHECK(cudaEventRecord(ev_start));
                kernel_seird_step<<<grid, block>>>(d_state_a, d_timer_a, d_state_b, d_timer_b, d_rng, N, params);
                kernel_count_states<<<grid, block>>>(d_state_b, N, d_counts);
                kernel_render_bgra<<<grid, block>>>(d_state_b, d_pixels, N);
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));

                float step_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&step_ms, ev_start, ev_stop));

                int h_counts[5] = {0};
                CUDA_CHECK(cudaMemcpy(h_counts, d_counts, 5 * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_pixels.data(), d_pixels, total_cells * sizeof(uint32_t), cudaMemcpyDeviceToHost));

                csv << day << "," << h_counts[0] << "," << h_counts[1] << ","
                    << h_counts[2] << "," << h_counts[3] << "," << h_counts[4] << "\n";

                char title[256];
                std::snprintf(title, sizeof(title),
                              "SEIRD CUDA (%dx%d) | Dia %d | S:%d E:%d I:%d R:%d D:%d | GPU: %.2f ms",
                              N, N, day, h_counts[0], h_counts[1], h_counts[2], h_counts[3], h_counts[4], step_ms);
                SetWindowTextA(hwnd, title);

                RECT client{};
                GetClientRect(hwnd, &client);
                StretchDIBits(hdc, 0, 0, client.right, client.bottom,
                              0, 0, N, N, h_pixels.data(), &bmi, DIB_RGB_COLORS, SRCCOPY);

                std::swap(d_state_a, d_state_b);
                std::swap(d_timer_a, d_timer_b);
            }

            auto elapsed = std::chrono::steady_clock::now() - frame_start;
            if (elapsed < frame_period) {
                std::this_thread::sleep_for(frame_period - elapsed);
            }
        }
        ReleaseDC(hwnd, hdc);
    }

    CUDA_CHECK(cudaFree(d_state_a));
    CUDA_CHECK(cudaFree(d_state_b));
    CUDA_CHECK(cudaFree(d_timer_a));
    CUDA_CHECK(cudaFree(d_timer_b));
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
