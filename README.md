# 🎓 Probabilistic Machine Learning Seminar  

### 📚 About  
This repository contains the code and experiments for my seminar thesis in the **Probabilistic Machine Learning** course (SoSe 2025).  
The work focuses on exploring **epistemic uncertainty** in neural networks using **partial Bayesian inference**, with a particular emphasis on:

- **Bayesian last-layer models** (Neural Linear Approximations)  
- Comparison to fully deterministic neural networks  
- Evaluation of **uncertainty calibration** and **overconfidence** on in-distribution and out-of-distribution (OOD) data

---

### 🛠️ Core Topics
- Probabilistic modeling in neural networks  
- Bayesian linear regression as output layer  
- Predictive entropy and epistemic uncertainty  
- OOD detection via uncertainty quantification  
- Comparison to MAP-trained deterministic models

---

### 📁 Project Structure

- `src/` – Jupyter notebooks for experiments and visualizations  
  - `Simulation.ipynb` – 1D toy regression to visualize predictive uncertainty  
  - `RealData.ipynb` – Evaluation on real-world data 
  - `BayesianLLNN.py` - Implementation of Bayesian Last Layer
  - `baseNN.py` - base deterministic NN
   
- `models/` – Models 
  - `baseBayes.pth` – base Neural Network for Classfication
  - `basenn.pth` - Standard ReLU neural network  
  - `best_lastlayer.pth` - – Neural Linear Model (Bayesian output layer)

- `Data/` – Data
  - `Dataset of Diabetes.csv` - Dataset in csv format
  
- `results/` – Generated plots and saved evaluation results

- `requirements.txt` – Python dependencies

- `README.md` – This file

---

### 📊 Datasets Used
- **Synthetic (toy) regression dataset** — for visualizing predictive uncertainty  
- **Real-world dataset**
  - Diabetes Diagnosis: available at [Kaggle](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset)
  - In-distribution: known label classes  
  - OOD: unrelated or random inputs for uncertainty evaluation

---

### 📌 Goal of the Work
To evaluate whether a simple Bayesian approximation (only in the last layer) is sufficient to capture meaningful uncertainty in predictions — particularly in OOD settings — and to compare its behavior to a fully deterministic ReLU network.

---

### LICENSE
This project is **not open-source**. All rights reserved. More infos found on [License](LICENSE) 
Unauthorized use, copying, or distribution is prohibited.



