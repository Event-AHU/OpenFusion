## HFPINNs 

* **Revisiting Heat Flux Analysis of Tungsten Monoblock Divertor on EAST using Physics-Informed Neural Network**,
  Xiao Wang, Zikang Yan, Hao Si, Zhendong Yang*, Qingquan Yang*, Dengdi Sun, Wanli Lyu, Jin Tang
  [[Paper](https://arxiv.org/abs/2508.03776)] 


### Abstract 
Estimating heat flux in the nuclear fusion device EAST is a critically important task. Traditional scientific computing methods typically model this process using the Finite Element Method (FEM). However, FEM relies on grid-based sampling for computation, which is computationally inefficient and hard to perform real-time simulations during actual experiments. Inspired by artificial intelligence-powered scientific computing, this paper proposes a novel Physics-Informed Neural Network (PINN) to address this challenge, significantly accelerating the heat conduction estimation process while maintaining high accuracy. Specifically, given inputs of different materials, we first feed spatial coordinates and time stamps into the neural network, and compute boundary loss, initial condition loss, and physical loss based on the heat conduction equation. Additionally, we sample a small number of data points in a data-driven manner to better fit the specific heat conduction scenario, further enhancing the model's predictive capability. We conduct experiments under both uniform and non-uniform heating conditions on the top surface. Experimental results show that the proposed thermal conduction physics-informed neural network achieves accuracy comparable to the finite element method, while achieving \times40 times acceleration in computational efficiency. 

<img width="1828" height="1154" alt="image" src="https://github.com/user-attachments/assets/4053cf54-013a-43e8-9acf-95657c089362" />



### Get Started

**Set up the environment**

```shell
conda create --name EAST python==3.8
conda activate EAST
pip install -r requirements.txt 
```

**Quick Start**

```shell
sh run.sh
```


### Experimental Results 

<img width="1710" height="1518" alt="image" src="https://github.com/user-attachments/assets/a0ea24f2-5309-4771-a54e-413069d5bd66" />



### Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base:

https://github.com/thuml/RoPINN

### Citation 
If you find this work useful for your research, please give us a star ⭐! 
```
@misc{wang2025revisitingheatfluxanalysis,
      title={Revisiting Heat Flux Analysis of Tungsten Monoblock Divertor on EAST using Physics-Informed Neural Network}, 
      author={Xiao Wang and Zikang Yan and Hao Si and Zhendong Yang and Qingquan Yang and Dengdi Sun and Wanli Lyu and Jin Tang},
      year={2025},
      eprint={2508.03776},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.03776}, 
}
```


