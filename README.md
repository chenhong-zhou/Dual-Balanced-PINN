# Dual-Balanced PINN (DB-PINN) 
Official code for the IJCAI 2025 paper "Dual-Balancing for Physics-Informed Neural Networks".



## Usage

Here we give the code to reproduce the results on six PDEs: Klein-Gordon Equation, Wave Equation, Helmholtz Equation, Allen-Cahn Equation, Burgers Equation, and Navier-Stokes Equation. 

Code was implemented in `python 3.7`. 

Just run the code in the .py script in the desired folder, for example:

```
python Klein-Gordon/Klein-Gordon.py
```

The training will begin, followed by the plots.



## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{zhou2025dual,
  title={Dual-Balancing for Physics-Informed Neural Networks},
  author={Chenhong Zhou and Jie Chen and Zaifeng Yang and Ching Eng Png},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}
```


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/Geo-FNO

https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/tree/master

https://github.com/cvjena/GradStats4PINNs

https://github.com/levimcclenny/SA-PINNs

