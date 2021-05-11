# MP-NeRF: Massively Parallel Natural Extension of Reference Frame for Efficient Internal to Cartesian Conversion

This is the code for the paper "Massively Parallel Natural Extension of Reference Frame for Efficient Internal to Cartesian Conversion"

## Abstract

The conversion of polymers between internal and cartesian coordinates is a limiting step in many pipelines, such as molecular dynamics simulations and training of machine learning models. This conversion is typically carried out by sequential or parallel applications of the Natural extension of Reference Frame (NeRF)algorithm. 

This work proposes a massively parallel NeRF implementation, which, depending on the polymer length, achieves speedups between 300-1200x over the most recent parallel NeRF implementation by dviding the conversion into three main phases: a parallel composition of the minimal repeated structure, the assembly of backbone subunits and the parallel elongation of sidechains. 

Special emphasis is placed on reusability and ease of use within diverse pipelines. We open source the code (available at https://github.com/EleutherAI/mp_nerf) and provide a corresponding python package.


## Results: 

* **Tests**: n an intel i5 at 2.4 ghz and nvidia 1060 gb

length   |  sota  | **us (cpu)** |  Nx   | us (gpu) 
---------|--------|--------------|-------|-----------
114      | 1.17s  | **5.97ms**   | ~400  | 18.6ms   
300      | 3.5s   | **8.60ms**   | ~405  | 22.6ms    
500      | 7.5s   | **11.65ms**  | ~630  | 26.5ms      
1000     | 18.66s | **18.21ms**  | ~1024 | 29.89ms      

* **Profiler Trace (CPU)**:
<center><img src="experiments/profiler_capture.png"></center>
<center><img src="experiments/histogram_errors.png"></center>
<center><img src="experiments/error_evolution.png"></center>

#### Considerations

* In the GPU algo, much of the time is spent in the data transfers / loop in the GPU is very inefficient. 
* about 1/2 of time is spent in memory-access patterns and the sequential `for loop`, so ideally 2x from here would be possible by optimizing it or running the sequential loop in cython / numba / whatever
* total profiler time should be multiplied by 0.63-0.5 to see real time (see execution above without profiler). Profiling slows down the code.


## Installation:

Just clone the repo

You'll need:
* torch > 1.6
* numpy
* einops

Plus, if you want to run the experiments / work with data: 
* joblib
* sidechainnet: https://github.com/jonathanking/sidechainnet#installation
* manually install `ProDY`, `py3Dmol`, `snakeviz`:
	* `pip install proDy`
	* `pip install py3Dmol`
	* `pip install snakeviz`
	* any other package: `pip install package_name`


* matplotlib (to do diagnostic plots)

## Citations:

```bibtex
@article{Parsons2005PracticalCF,
    title={Practical conversion from torsion space to Cartesian space for in silico protein synthesis},
    author={Jerod Parsons and J. B. Holmes and J. M. Rojas and J. Tsai and C. Strauss},
    journal={Journal of Computational Chemistry},
    year={2005},
    volume={26}
}
```

```bibtex
@article{AlQuraishi2018pNeRFPC,
    title={pNeRF: Parallelized Conversion from Internal to Cartesian Coordinates},
    author={Mohammed AlQuraishi},
    journal={bioRxiv},
    year={2018}
}
```

```bibtex
@article{Bayati2020HighperformanceTO,
    title={High‐performance transformation of protein structure representation from internal to Cartesian coordinates},
    author={M. Bayati and M. Leeser and J. Bardhan},
    journal={Journal of Computational Chemistry},
    year={2020},
    volume={41},
    pages={2104 - 2114}
}
```

