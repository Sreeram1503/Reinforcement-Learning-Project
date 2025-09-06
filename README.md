# Deep Learning for Autonomous Control in a Simulated Racing Environment

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

This project leverages deep learning to optimize the driving performance of an AI agent in the PySuperTuxKart environment, with the primary goals of reducing track completion times and optimizing the training pipeline.

### Key Features
- **Advanced CNN Planner:** A Convolutional Neural Network with a spatial soft-argmax layer predicts optimal aim-points directly from image frames, simplifying the training process.
- **Adaptive Controller:** A sophisticated, non-learning controller with dynamic logic for steering, acceleration, braking, and drifting, designed to navigate complex track curvatures.
- **Iterative Training & Refinement:** A multi-stage approach was used to progressively improve the training data and model architecture, leading to significant performance gains.
- **Reinforcement Learning Exploration:** Includes experimental implementation of a Proximal Policy Optimization (PPO) agent using the Gymnasium framework.

---

### Technical Approach & Methodology

The solution was developed in three main stages:

**1. Baseline Controller & Initial CNN:**
- A baseline controller with linear steering and constant acceleration was developed to generate an initial dataset.
- An initial 3-layer CNN was trained using Mean Squared Error (MSE) loss, which plateaued at a loss of 0.145, highlighting the need for more sophisticated data and models.

**2. Advanced CNN & Training Pipeline:**
- The CNN architecture was enhanced with sequential 5x5 convolutional blocks and a spatial soft-argmax layer for more precise aim-point prediction.
- The training process was upgraded by switching to L1 loss for robustness to outliers and implementing data augmentation techniques (color variability, random horizontal flips).

**3. Dynamic Controller & Final Integration:**
- The final controller incorporated adaptive mechanisms, including exponential steering logic for smoother control, dynamic braking based on lateral offsets, and strategic use of nitro/drifting.
- This refined controller, paired with the advanced CNN, achieved a final training loss of **0.0245**, demonstrating a highly effective planner.

---

### Results

The final planner (Stage 3) showed significant improvements in track completion times compared to earlier stages, performing on par with the highly tuned adaptive controller.

| Track | Stage 1 Planner | Stage 2 Planner | Stage 3 Planner |
| :--- | :---: | :---: | :---: |
| **Zengarden** | 654s | 498s | 397s |
| **Lighthouse** | 689s | 464s | 434s |
| **Hacienda** | - | 554s | 533s |
| **SnowTuxPeak** | - | 580s | 472s |
| **Cornfield Crossing** | - | 734s | 686s |
| **Scotland** | - | 642s | 568s |

*(Completion times are in seconds. Lower is better.)*

#### Demo Videos
- [Link to Test Video 1](https://drive.google.com/file/d/1p2Ulju4LxvDfASDrUcQjwgtdWaOTUYc-/view?usp=sharing)
- [Link to Test Video 2](https://drive.google.com/file/d/1u6YteG1maRnY_PhlPGoAagj7anLe_5hV/view?usp=sharing)

---

### Technologies Used
- **Primary Language:** Python
- **Key Libraries:** PyTux, Gymnasium, stable-baselines3
- **ML/DL Frameworks:** PyTorch (or your relevant framework)
- **Concepts:** Deep Learning, Reinforcement Learning, Convolutional Neural Networks (CNNs), Computer Vision, Optimization

---

### Setup & Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Sreeram1503/EC418](https://github.com/Sreeram1503/EC418)
    ```
2.  Install the required dependencies (you may need to create a `requirements.txt` file).
3.  Run the planner or controller using the provided utility scripts.
