
#include "kernelmatrix.h"

#include <stdio.h>

#include "basic.h"
#include "h2compression.h"
#include "h2matrix.h"
#include "matrixnorms.h"
#include "parameters.h"

static field
kernel_newton(const real *xx, const real *yy, void *data)
{
    real norm2;

    (void) data;

    norm2 = REAL_SQR(xx[0] - yy[0]) + REAL_SQR(xx[1] - yy[1]) + REAL_SQR(xx[2] - yy[2]);

    return (norm2 == 0.0 ? 0.0 : 1.0 / REAL_SQRT(norm2));
}

static field
kernel_exp(const real *xx, const real *yy, void *data)
{
    real norm2;

    (void) data;

    norm2 = REAL_SQR(xx[0] - yy[0]) + REAL_SQR(xx[1] - yy[1]) + REAL_SQR(xx[2] - yy[2]);

    return REAL_EXP(-norm2);
}

static field
kernel_log(const real *xx, const real *yy, void *data)
{
    real norm2;

    (void) data;

    norm2 = REAL_SQR(xx[0] - yy[0]) + REAL_SQR(xx[1] - yy[1]) + REAL_SQR(xx[2] - yy[2]);

    return (norm2 == 0.0 ? 0.0 : -0.5*REAL_LOG(norm2));
}

int
main(int argc, char **argv)
{
    pkernelmatrix km;
    pclustergeometry cg;
    pcluster root;
    pblock broot;
    pclusterbasis cb;
    ph2matrix Gh1, Gh2;
    pamatrix G;
    pavector x, y0, y1;
    pstopwatch sw;
    char kernel;
    uint points;
    uint m, leafsize;
    uint *idx;
    size_t sz;
    real eta, mindiam;
    real eps;
    real t_setup, norm, error;
    real t_matvec, y0_2norm, ydiff_2norm, diff;
    uint i;

    // ========== Initialize test parameters ==========

    init_h2lib(&argc, &argv);

    sw = new_stopwatch();

    kernel = askforchar("Kernel function? N)ewton, L)ogarithmic, or E)xponential?", "h2lib_kernelfunc", "nle", 'n');

    points = askforint("Number of points?", "h2lib_kernelpoints", 2048);

    m = askforint("Interpolation order?", "h2lib_interorder", 2);

    leafsize = askforint("Cluster resolution?", "h2lib_leafsize", 2*m*m);

    eps = askforreal("Recompression tolerance?", "h2lib_comptol", 1e-4);

    eta = 2.0;
    
    (void) printf("Creating kernelmatrix object for %u points, order %u\n", points, m);
    km = new_kernelmatrix(3, points, m);
    switch(kernel) {
    case 'e':
        (void) printf("    Exponential kernel function\n");
        km->kernel = kernel_exp;
        break;
    case 'n':
        (void) printf("    Newton kernel function\n");
        km->kernel = kernel_newton;
        break;
    default:
        (void) printf("    Logarithmic kernel function\n");
        km->kernel = kernel_log;
    }
    for(i=0; i<points; i++) {
        km->x[i][0] = FIELD_RAND();	/* Random points in [-1,1]^2 */
        km->x[i][1] = FIELD_RAND();
        km->x[i][2] = FIELD_RAND();
    }

    // ========== Initialize partitioning tree ==========

    (void) printf("Creating clustergeometry object\n");
    cg = creategeometry_kernelmatrix(km);

    (void) printf("Creating cluster tree\n");
    idx = (uint *) allocmem(sizeof(uint) * points);
    for (i=0; i<points; i++) idx[i] = i;
    root = build_adaptive_cluster(cg, points, idx, leafsize);
    (void) printf("    %u clusters, depth %u\n", root->desc, getdepth_cluster(root));
    
    (void) printf("Creating block tree\n");
    broot = build_strict_block(root, root, &eta, admissible_2_cluster);
    (void) printf("    %u blocks, depth %u\n", broot->desc, getdepth_block(broot));

    (void) printf("Creating cluster basis\n");
    cb = build_from_cluster_clusterbasis(root);

    (void) printf("Filling cluster basis\n");
    start_stopwatch(sw);
    fill_clusterbasis_kernelmatrix(km, cb);
    t_setup = stop_stopwatch(sw);
    sz = getsize_clusterbasis(cb);
    (void) printf("    %.2f seconds\n"
		"    %.1f MB\n"
		"    %.1f KB/DoF\n",
		t_setup, sz / 1048576.0, sz / 1024.0 / points);

    // ========== Build H2 matrix ==========

    start_stopwatch(sw);
    (void) printf("Creating H^2-matrix (Gh1) \n");
    Gh1 = build_from_block_h2matrix(broot, cb, cb);
    sz = getsize_h2matrix(Gh1);
    (void) printf("Gh1: %.1f MB\n"
		"     %.1f KB/DoF\n",
		sz / 1048576.0, sz / 1024.0 / points);
    (void) printf("Filling H^2-matrix (Gh1) \n");
    
    fill_h2matrix_kernelmatrix(km, Gh1);

    (void) printf("Recompressing H^2-matrix (Gh2), eps=%g\n", eps);
    Gh2 = compress_h2matrix_h2matrix(Gh1, false, false, 0, eps);
    t_setup = stop_stopwatch(sw);
    sz = getsize_h2matrix(Gh2);
    (void) printf("Gh1 & Gh2: %.2f seconds\n"
		"Gh2: %.1f MB\n"
		"     %.1f KB/DoF\n",
		t_setup, sz / 1048576.0, sz / 1024.0 / points);
    
    // ========== Test H2 matvec ==========
    x  = new_avector(points);
    y0 = new_avector(points);
    y1 = new_avector(points);
    random_real_avector(x);
    
    clear_avector(y1);
    addeval_h2matrix_avector(1.0, Gh2, x, y1);
    t_matvec = 0;
    (void) printf("Performing 10 times H2 matvec\n");
    for (i = 0; i < 10; i++)
    {
        clear_avector(y1);
        start_stopwatch(sw);
        addeval_h2matrix_avector(1.0, Gh2, x, y1);
        t_matvec += stop_stopwatch(sw);
    }
    (void) printf("    H2 matvec average time = %.3lf s\n", t_matvec / 10.0);
    
    // ========== Check approximation error ==========
    
    (void) printf("Filling reference matrix\n");
    G = new_amatrix(points, points);
    start_stopwatch(sw);
    fillN_kernelmatrix(0, 0, km, G);
    t_setup = stop_stopwatch(sw);
    sz = getsize_amatrix(G);
    (void) printf("    %.2f seconds\n"
		"    %.1f MB\n"
		"    %.1f KB/DoF\n",
		t_setup, sz / 1048576.0, sz / 1024.0 / points);

    /*
    (void) printf("Computing reference matrix norm\n");
    norm = norm2_amatrix(G);
    (void) printf("    Reference matrix spectral norm %.3e\n", norm);

    (void) printf("Computing Gh1 norm\n");
    norm = norm2_h2matrix(Gh1);
    (void) printf("    Gh1 spectral norm %.3e\n", norm);

    (void) printf("Computing Gh1 approximation error\n");
    error = norm2diff_amatrix_h2matrix(Gh1, G);
    (void) printf("    Gh1 spectral error %.3e (%.3e)\n", error, error/norm);

    (void) printf("Computing Gh2 approximation error\n");
    error = norm2diff_amatrix_h2matrix(Gh2, G);
    (void) printf("    Gh2 spectral error %.3e (%.3e)\n", error, error/norm);
    */
    
    (void) printf("Comparing reference matrix matvec & H2 matvec\n");
    clear_avector(y0);
    addeval_amatrix_avector(1.0, G, x, y0);
    y0_2norm = 0;
    ydiff_2norm = 0;
    for (i = 0; i < points; i++)
    {
        diff = y0->v[i] - y1->v[i];
        y0_2norm += y0->v[i] * y0->v[i];
        ydiff_2norm += diff * diff;
    }
    y0_2norm = REAL_SQRT(y0_2norm);
    ydiff_2norm = REAL_SQRT(ydiff_2norm);
    (void) printf("    ||y_{H2} - y_{ref}||_2 / ||y_{ref}||_2 = %e\n", ydiff_2norm / y0_2norm);

    uninit_h2lib();

    return 0;
}
