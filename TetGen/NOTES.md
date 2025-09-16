# Benchmarking TetGen agains DTCC Volume Mesher

## Test domain

500m x 500m x 100m box centered at Poseidon:

    x0 = 319995.962899
    y0 = 6399009.716755
    L = 500.0
    H = 100.0
    bounds = dtcc.Bounds(x0 - 0.5 * L, y0 - 0.5 * L, x0 + 0.5 * L, y0 + 0.5 * L)

## Metrics

N = Number of cells
T = CPU time, measured in terminal via time, using last column (total)
Q = Mesh quality (aspect ratio in Paraview)

## TetGen

## #Cells CPU time Quality

V = max_tet_volume

Max coarse: 1e9 191,003 7.018 11.900

V N T Q

20000.0 187,212 7.045 9.501
8000.0 191,131 7.139 8.199
4000.0. 194,178 6.999 11.881
2000.0 199,630 7.365 12.394
800.0 246,066 7.450 10.127
400.0 319,850 8.067 11.420
200.0 454,080 8.608 11.471
80.0 860,494 9.696 12.657
40.0 1,491,720 12.270 11.528
20.0 2,738,396 16.741 13.134
10.0 5,202,535 26.267 17.682
5.0 10,020,374 51.453 12.538
2.5 19,476,910 97.49 13.300
1.0 47,664,041 248.04 22.119

## DTCC

h = max_mesh_size

## #Cells CPU time Quality

h N T Q

Max coarse: 200.0 4,146,784 71.0 1e30
Max coarse: 80.0 4,146,784 61.25 1e30

40.0 4,146,784 69.44 1e30
20.0 2,646,227 8.735 1e30
10.0 1,431,408 24.495 16.027
5.0 1,687,646 31.053 7377.027
2.5 8,308,283 121.73 875.932
1.0 STOPPED
