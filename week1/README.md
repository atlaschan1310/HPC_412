Exercise 1:
blitz -> namespace
Array<float, 3> -> template
data(10, 8, 6) -> constructor
This statements actually creates a 3-dimensional array called data in which all elements are float-type. The dimensions of this array is 10, 8, 6, respectively.
I am not sure about the intension of the other part of Exercise 1, please check my code. 

Exercise 2:
numberOfNodes = ceil(4096^3 * 56 / (1000^3 * 64)) = 61

Exercise 3:
Simulation time: 1.0000000000000107 
god -N 8 -t f8 --endian=big b0-final.std
Number of Particles: 158095
god -N 4 -t d4 --endian=big -j 8 b0-final.std

Exercise 4:
mass: 2.0393363e-09
x, y, z: -0.008689348     -0.03393134     -0.03598262
offset = 32 + 99 * 36 = 3596
god -N 16 -t f4 --endian=big -j 3596 b0-final.std

Exercise 5:
xmin:  -0.499911904335022 xmax:  0.4999992549419403
ymin:  -0.4998406171798706 ymax:  0.49973100423812866
zmin:  -0.49995994567871094 zmax:  0.49990418553352356
total mass: 0.23699994

