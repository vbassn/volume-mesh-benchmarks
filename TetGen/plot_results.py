from pylab import *
from matplotlib import style

style.use("dark_background")

tetgen = [
    [187212, 7.045, 9.501],
    [191131, 7.139, 8.199],
    [194178, 6.999, 11.881],
    [199630, 7.365, 12.394],
    [246066, 7.450, 10.127],
    [319850, 8.067, 11.420],
    [454080, 8.608, 11.471],
    [860494, 9.696, 12.657],
    [1491720, 12.270, 11.528],
    [2738396, 16.741, 13.134],
    [5202535, 26.267, 17.682],
    [10020374, 51.453, 12.538],
    [19476910, 97.49, 13.300],
    [47664041, 248.04, 22.119],
]

dtcc = [
    [4146784, 69.44, 1e30],
    [2646227, 8.735, 1e30],
    [1687646, 31.053, 7377.027],
    [1431408, 24.495, 16.027],
    [8308283, 121.73, 875.932],
]

tetgen = array(tetgen)
dtcc = array(dtcc)

figure(figsize=(8, 10))

subplot(2, 1, 1)
plot(tetgen[:, 0], tetgen[:, 1], "o-", label="TetGen")
plot(dtcc[:, 0], dtcc[:, 1], "o-", label="DTCC")
grid(True)
legend()
title("Volume mesh generation performance")
# xlabel("Number of tetrahedra")
ylabel("Time (s)")
xscale("log")
yscale("log")

subplot(2, 1, 2)
plot(tetgen[:, 0], tetgen[:, 2], "o-", label="TetGen")
# plot(dtcc[:, 0], dtcc[:, 2], "o-", label="DTCC")
grid(True)
legend()
title("Volume mesh generation quality")
xlabel("Number of tetrahedra")
ylabel("Quality (aspect ratio)")
xscale("log")
# yscale("log")

show()
