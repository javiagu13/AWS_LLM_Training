# Train LLMs in Large AWS Clusters

This repository was created to address the lack of open-source tools for training Large Language Models (LLMs) in large AWS clusters. AWS provides minimal support for this use case, and existing solutions often fall short in meeting the technical challenges researchers face. This repository bridges that gap, offering a practical and tested solution.

---

## 🚀 Features

- **AWS Cluster Compatibility**: Designed for AWS `p4d` and `p4de` clusters, each equipped with 8 A100 GPUs.
  - `p4d`: Each GPU provides **40GB** memory.
  - `p4de`: Each GPU provides **80GB** memory.
- **Supported Model Sizes**: Successfully tested on models ranging from **8B** to **13B** parameters, including:
  - LLaMA 3.1 (8B)
  - Gemma 2 (7B)
  - KullM3 (10.7B)
  - LG Exaone 3.5 (7B)
- **Performance**:
  - Average training time:
    - **p4d**: ~1 hour (with gradient offloading to CPU due to memory constraints).
    - **p4de**: ~30 minutes (with full CUDA-based training).
  - Memory usage:
    - **p4d**: ~37GB per GPU.
    - **p4de**: ~47GB per GPU (higher due to reduced offloading needs).

---

## 🛠️ Challenges Addressed

Training LLMs on AWS clusters comes with unique technical challenges. This repository tackles key obstacles, including:

1. **Integration with AWS S3**:
   - Running training jobs (cost-efficient compared to notebook instances) requires using AWS native libraries, which can lead to compatibility conflicts.
   
2. **Containerized Training**:
   - Since training jobs are executed inside containers, the code must be tailored to work seamlessly within this environment.

3. **Parallelization Issues**:
   - Enabling tools like DeepSpeed to handle parallelization (e.g., model, gradient, and optimizer sharding) was a significant hurdle.
   - While 8 A100 GPUs may seem sufficient, training in mixed precision (BF16) poses challenges due to GPU memory limits. Native PyTorch techniques like `DataParallel` and `ModelParallel` often crash due to excessive GPU consumption, even in large clusters.

---

## 🧠 Our Approach

We leverage **DeepSpeed** to handle the complex requirements of parallelization and optimization, ensuring efficient utilization of GPU memory and computational power. By striking the right balance between speed and resource use, our method proves to be more economical than using slower, cheaper clusters.

### Key Components
This repository includes three essential notebooks to streamline the workflow:

1. **AWS Training Notebook**: Configures and executes the training job in an AWS environment.
2. **Dataset Preprocessing Notebook**: Prepares datasets for training.
3. **S3 to Hugging Face Export Notebook**: Simplifies the process of exporting trained models from S3 to Hugging Face.

---

## 📊 Why This Repository?

Despite the high cost of AWS clusters, the speed of training achieved using this repository makes it more economical compared to prolonged use of cheaper clusters. By overcoming critical technical barriers, this repository provides a robust, reusable solution for researchers and engineers working on state-of-the-art LLMs.

---

## 🏆 Tested Models and Research Use Cases

This repository has been extensively tested and used for training models in the **8B–13B parameter range** across various research projects. Its effectiveness and versatility make it an invaluable tool for anyone working in this space.

---

## 🛡️ Disclaimer

To use this repository effectively, familiarity with AWS services (e.g., S3, EC2, training jobs) and deep learning tools (e.g., PyTorch, DeepSpeed) is recommended.

