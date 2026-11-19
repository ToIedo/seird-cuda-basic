# SEIRD-CUDA-Basic

Versión minimalista, visual y didáctica del modelo epidemiológico **SEIRD** sobre una **grilla 2D (Autómata Celular Estocástico)** en **CUDA**, diseñada para ejemplificar el desarrollo de kernels paralelos en GPU.

## Estructura del proyecto

- `src/seird_basic.cu`: Código fuente completo en un solo archivo (3 kernels CUDA + ventana visual Win32/GDI en tiempo real + modo benchmark CPU vs GPU).
- `build.bat`: Script de compilación para Windows con Visual Studio 2022 Build Tools + `nvcc` (CUDA 13.5, arquitectura `sm_120` para RTX 5050).
- `output/seird_curve.csv`: Evolución diaria de `S, E, I, R, D`.

## Compilación y ejecución (Windows)

```bat
build.bat
build\seird_basic.exe
```

Al ejecutar sin argumentos abre la **ventana visual en tiempo real** (`512x512` a `30 FPS`):
- `Espacio`: Pausar / Reanudar
- `R`: Reiniciar simulación
- `Esc`: Salir

Opciones por línea de comandos:
```bat
build\seird_basic.exe --size 1024 --fps 60
build\seird_basic.exe --bench --size 1024 --days 200
```

## Kernels CUDA incluidos

1. `kernel_seird_step`: Mapeo 2D (`1 hilo = 1 celda`), vecindad de Moore (8 vecinos), doble buffer (`ping-pong`) y PRNG `xorshift32`.
2. `kernel_count_states`: Conteo paralelo usando memoria compartida (`__shared__`) y `atomicAdd`.
3. `kernel_render_bgra`: Conversión directa en GPU de los estados `S, E, I, R, D` a píxeles `BGRA` para mostrar en pantalla.

## Código de colores

- **Susceptible (S)**: Azul oscuro
- **Expuesto (E)**: Amarillo
- **Infectado (I)**: Rojo
- **Recuperado (R)**: Verde
- **Fallecido (D)**: Gris oscuro
