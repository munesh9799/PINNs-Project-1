# sdof-pinns
Single-Degree-Of-Freedom (SDOF) systems, despite their apparent simplicity, are fundamental building blocks for analyzing and interpreting the dynamic behavior of more complex multi-degree-of-freedom (MDOF) structures. Developing efficient and robust methodologies for both forward and inverse modeling of these systems is essential for various engineering applications, including structural design verification and structural health monitoring.

In this project, we introduce the use of Physics-Informed Neural Networks (PINNs) for both the forward and inverse dynamic analyses of SDOF systems. In the forward formulation, our goal is to predict the structural response in terms of displacement, velocity, and acceleration based on known physical parameters and external excitations. Conversely, the inverse formulation aims to estimate unknown stiffness parameters using measured dynamic responses.

By incorporating the governing differential equations of motion directly into the loss function, the PINN framework ensures compliance with physical laws while leveraging the approximation capabilities of deep neural networks. 

This repository contains PyTorch implementations of Physics-Informed Neural Networks (PINNs) for:
- **Forward modeling** of an underdamped single-degree-of-freedom (SDOF) free-vibration system.
- **Inverse modeling** (parameter identification) to estimate **ω** from noisy displacement observations.
- **Scaled forward modeling** using an ansatz with learnable sinusoidal modulation to handle higher frequencies (ω = 80).

## Governing equation
\[
\frac{d^2 u(t)}{dt^2} + 2\,\xi\,\omega\,\frac{d u(t)}{dt} + \omega^2\,u(t) = 0,
\]

subject to the initial conditions

\[
u(0) = u_0, \qquad \frac{d u(0)}{dt} = v_0.
\]

This work has been presented at the 4th International Conference on Sustainable Development in Civil Engineering, 2025. For citation, refer to the following:
## Citation

```bibtex
@inproceedings{Meghwar2025PINNsSDOF_Abstract,
  title     = {Forward and Inverse Modeling of SDOF Structures Using Physics-Informed Neural Networks (PINNs)},
  author    = {Meghwar, Munesh and Meghwar, Shanker Lal and Ansari, Jawaid Kamal and Sahito, Abdul Munim and Rahu, Sheeraz Ahmed},
  booktitle = {Abstracts of the 4th International Conference on Sustainable Development in Civil Engineering},
  year      = {2025},
  address   = {Pakistan},
  publisher = {Mehran University of Engineering \& Technology, Jamshoro},
  note      = {Conference abstract}
}


