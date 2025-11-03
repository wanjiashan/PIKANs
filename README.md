## PIKANs: Physics-Informed Kolmogorov-Arnold Networks for Landslide Time-to-Failure Prediction

### Abstract

> *Slope deformation is characterized by pronounced time variability and complexity. Although ground-based synthetic aperture radar (GB-SAR) provides high-frequency, broad monitoring, its strong oscillations and large fluctuations can impair predictive performance. To address this, the raw displacement sequence is first smoothed via misaligned subtraction to suppress high-frequency noise and highlight key deformation trends. A dynamic confidence boundary is then established on the inverse-velocity curve to robustly identify the acceleration start point. Building on prior work on physics-informed Kolmogorov-Arnold networks (PIKANs), we apply a PIKANs framework to landslide early warning, embedding the displacement-time evolution constraint into the basis-function space of Kolmogorov-Arnold network (KAN) to unify nonlinear deformation dynamics with governing physical laws. During model training, an alternating optimization scheme combining Adam and the L-BFGS algorithm accelerates convergence and enhances predictive accuracy. Comparative experiments on field GB-SAR datasets demonstrate that compared with an improved KAN baseline and a physics-informed neural network benchmark, PIKANs reduce the relative error in failure-time prediction by 38.42% and 20.44%, respectively. These results confirm that integrating physical equation constraints into neural network parameter updates substantially improves the precision and efficiency of real-time landslide early warning.*
<div align="center">
  <img src="imgs/1.png" alt="Example Image" width="500" />
</div>


## RUN
- Python 3.9 – 3.11
- numpy
- pandas
- openpyxl
- torch (CPU 或 CUDA 版本)
- python pikan.py












