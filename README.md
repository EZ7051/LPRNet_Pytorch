# Real-Time AI Model Optimization

This project focuses on optimizing and deploying a real-time AI model for License Plate Recognition (LPR). The pipeline includes model pruning, quantization, and graph-level optimizations using TVM, AutoTVM, and fine-tuning techniques to achieve reduced inference time and model size while maintaining accuracy.

---

## Steps to Run the Project

1. **Upload Notebook**:
   - Upload the file `FinalRealTimeAI.ipynb` to Google Colab.

2. **Clone the Original LPRNet Repository**:
   - Run the **first cell** in the notebook to clone the original LPRNet GitHub repository.

3. **Replace `load_data.py` File**:
   - After cloning, replace the `LPRNET_PYTORCH/data/load_data.py` file in the cloned repository with the updated version.
   - The updated `load_data.py` can be found in this GitHub repo.

4. **Run All Cells**:
   - Run all cells in the notebook **in order** to execute the full pipeline.

5. **View Results**:
   - Results, including inference times, model size reductions, and accuracy metrics, can be viewed directly in the Colab outputs.

---

## Key Features

1. **Pruned Model**:
   - Applied unstructured pruning with a pruning ratio of 0.4, reducing non-zero parameters and achieving a smaller model size and faster inference time.

2. **Graph-Level Optimizations**:
   - Four optimizations were applied:
     - **SimplifyInference**: Removed redundant operations during inference.
     - **FoldConstant**: Precomputed constant sub-expressions.
     - **FuseOps**: Combined multiple operations into a single kernel for better performance.
     - **AlterOpLayout**: Adjusted data layouts to leverage hardware capabilities effectively.

3. **AutoTVM Fine-Tuning**:
   - Fine-tuned 13 convolutional layers using `XGBTuner`, further improving inference time without accuracy degradation.

4. **Accuracy Retention**:
   - Maintained accuracy, dropping minimally from 89% to 87.1% through the optimization pipeline.

5. **Model Compression**:
   - Reduced model size significantly from 1.7 MB to 0.51 MB.

---

## Outputs
- **Inference Time**: Reduced from 218 ms to 30.79 ms.
- **Model Size**: Compressed from 1.7 MB to 0.51 MB.
- **Accuracy**: Retained at 87.1% across 1,000 test samples.

---


For further questions or issues, refer to the GitHub repository or contact the project contributors.
