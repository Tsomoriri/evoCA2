# evolutionary computing CA2
steps:

Step 1: LSTM and NAS
Search Algorithm: Long Short-Term Memory (LSTM)
Optimization Problem: Neural Architecture Search (NAS)

Step 2: Research LSTM and NAS

Study the fundamentals of LSTM, including its architecture, gating mechanisms, and its ability to capture long-term dependencies.
Research NAS, its objectives, search spaces, and various approaches (e.g., reinforcement learning, evolutionary algorithms, gradient-based methods).
Focus on recent literature that discusses the application of LSTM to NAS, as well as advancements and variations in both LSTM and NAS.
Organize your literature review by discussing the key aspects of LSTM and NAS, their strengths and limitations, and their combined use.

Step 3: Describe the problem formulation, search algorithm, and adaptations

Formulate the NAS problem, defining the search space (e.g., layer types, hyperparameters), objective function (e.g., accuracy, model complexity), and constraints (e.g., computational budget).
Explain how LSTM can be used as a controller to generate architectural configurations in the NAS search space.
Discuss how you will adapt LSTM for NAS:
Representation of architectures (e.g., sequences of layer types and hyperparameters)
Reward function for guiding the LSTM controller (e.g., validation accuracy, model size)
Training strategy for the LSTM controller (e.g., reinforcement learning, policy gradient)
Use diagrams to illustrate the NAS search space and the LSTM controller architecture.

Step 4: Present pseudocode for your algorithms

Provide high-level pseudocode for the LSTM-based NAS algorithm, including the main loop, architecture generation, and controller update.
Include pseudocode for the reward calculation, architecture evaluation, and any NAS-specific components.
Ensure the pseudocode is clear, well-structured, and properly commented. Use descriptive variable names to enhance readability.

Step 5: Design a comprehensive set of experiments

Define your experimental objectives (e.g., evaluate the impact of different reward functions, compare LSTM-based NAS with other NAS methods).
Select a range of datasets and tasks (e.g., image classification, language modeling) to test the generalization ability of your NAS approach.
Choose performance metrics (e.g., accuracy, model complexity, search efficiency) and statistical tests for comparison.
Determine the range of values for LSTM and NAS hyperparameters (e.g., LSTM hidden size, learning rate, search budget) to be tuned.
Plan experiments to compare your LSTM-based NAS with random search and a simple evolutionary NAS method on the chosen datasets and tasks.
Justify your choices and explain how the experiments will provide insights into the performance and behavior of LSTM-based NAS.
By following these steps, you'll have a solid plan for your assignment using LSTM and NAS. Remember to document your work thoroughly, provide clear explanations for your decisions, and cite relevant literature to support your approach. In the next steps, you'll focus on implementing the experiments, presenting the results, and analyzing the findings.





------------------------------------------------------------
Recent literature on LSTM and NAS:

Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. In International Conference on Learning Representations (ICLR).
Seminal work on NAS, using an LSTM-based controller to generate architectures and reinforcement learning to train the controller.
Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., & Dean, J. (2018). Efficient neural architecture search via parameter sharing. In International Conference on Machine Learning (ICML).
Introduced a more efficient NAS approach called ENAS, which shares parameters among child models to reduce search costs.
Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable architecture search. In International Conference on Learning Representations (ICLR).
Proposed a gradient-based NAS method called DARTS, which relaxes the search space to be continuous and enables joint optimization of architecture and weights.
Dong, X., & Yang, Y. (2019). Searching for a robust neural architecture in four GPU hours. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
Presented a NAS approach that uses an LSTM-based super network to generate architectures and a progressive search strategy to improve efficiency.
Ren, P., Xiao, Y., Chang, X., Huang, P. Y., Li, Z., Gupta, S., Chen, X., & Wang, X. (2020). A comprehensive survey of neural architecture search: Challenges and solutions. arXiv preprint arXiv:2006.02903.
A comprehensive survey of NAS methods, discussing challenges, solutions, and future directions, including the use of LSTM-based controllers.
