## PIKANs: Physics-Informed Kolmogorov-Arnold Networks for Landslide Time-to-Failure Prediction

lope deformation is characterized by pronounced time variability and complexity. Although ground-based synthetic aperture radar (GB-SAR) provides
high-frequency, broad monitoring, its strong oscillations and large fluctuations
can impair predictive performance. To address this, the raw displacement se￾quence is first smoothed via misaligned subtraction to suppress high-frequency
noise and highlight key deformation trends. A dynamic confidence boundary
is then established on the inverse-velocity curve to robustly identify the ac￾celeration start point. Subsequently, physics-informed Kolmogorov-Arnold net￾works (PIKANs) are formulated to embed the displacement-time evolution equation into the basis-function space of the Kolmogorov-Arnold network (KAN),
thereby unifying nonlinear deformation dynamics with the governing physical
laws of landslide motion. During model training, an alternating optimization
scheme combining Adam and the L-BFGS algorithm accelerates convergence
and enhances predictive accuracy. Comparative experiments on field GB-SAR
datasets demonstrate that compared with an improved KAN baseline and a
physics-informed neural network benchmark, PIKANs reduce the relative error
Email addresses: Corresponding author: jswanwo@gmail.com (Jiashan Wan),
1033348860@qq.com (Liangjun Wen), w124302075@stu.ahu.edu.cn (Ziheng Jian),
jhwu3@iflytek.com (Jinhua Wu), sakajy21@gmail.com (Jingyang Li), abzzorro@gmail.com
(Mengqi Lian), wangkai_anhui@foxmail.com (Kai Wang)
Preprint submitted to Computers & Geosciences September 2, 2025
in failure-time prediction by 24.24% and 30.23%, respectively. These results
confirm that integrating physical equation constraints into neural network parameter updates substantially improves the precision and efficiency of real-time
landslide early warning.
Keywords: Failure time of landslides, ground-based synthetic aperture radar,
physics-informed neural network, Kolmogorov-Arnold networks

<div align="center">
  <img src="imgs/1.png" alt="Example Image" width="500" />
</div>

<div align="center">
  <img src="imgs/2-3.png" alt="Example Image" width="500" />
</div>



##run
First, you need to compress the data set. For example, when running PEMSBY, adjust the parameters in prepare.py if speed_sequences.shape[2] > 325: speed_sequences = speed_sequences[:, :, :325] and the parameter mamba_features=325 in train_STGmamba. This corresponds to the characteristics of the specific data set. For example, PESMBY is 325, and metr-la is 207. You need to adjust it and run the code.
```bash
#PEMS04
  python main.py -dataset=pems04 -model=TFPredictor -mamba_features=307
```
```bash
#PESMSBY
  python main.py -dataset=PEMSBY -model=TFPredictor -mamba_features=325 
```
```bash
#metr-la
  python main.py -dataset=metr-la -model=TFPredictor -mamba_features=207
```


