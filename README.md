# Simple Transformers for Amazon Reviews Classification

This project aims to explore and understand the structures within transformer models and adaptation layers such as LoRA (Low-Rank Adaptation) and RoSA (Row-Sparse Adaptation). The main focus is on the classification of Amazon reviews, employing a simple transformer architecture to distinguish between positive and negative sentiments.

## Overview

Transformers have revolutionized the way we approach text processing tasks, offering significant improvements in understanding and generating human-like text. This project leverages transformer models to classify Amazon product reviews, further enhancing the model's adaptability and efficiency with LoRA and RoSA layers. The intention is to deepen the understanding of these complex structures and their practical applications.

## Features

- Utilizes a custom transformer architecture designed for the binary classification of Amazon reviews.
- Incorporates adaptation layers (LoRA and RoSA) to explore their effects on model performance and efficiency.
- Detailed exploration of the transformer architecture and adaptation mechanisms, providing insights into their operational principles and benefits.
- Documentation of challenges encountered due to computational constraints and proposed methodologies for overcoming these limitations.

## Dataset

The dataset used in this project is a collection of Amazon reviews, which can be found on [Kaggle: Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). This dataset offers a wide range of product reviews, facilitating a comprehensive analysis and classification task.

## References

- LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- RoSA: [Row-Sparse Attention Mechanism for Deep Transformer Models](https://arxiv.org/abs/2004.13922)

Please refer to these papers for an in-depth understanding of the adaptation layers implemented in this project.

## Future Work and Improvements

This project opens up several avenues for future research and development:

- **Scalability:** Testing the models on larger datasets and across more diverse review sets to assess scalability and robustness.
- **Multi-GPU Training:** Leveraging parallel computing resources to reduce training time and potentially improve model performance.
- **Hyperparameter Optimization:** Systematic exploration of learning rates, batch sizes, and adaptation layer configurations to optimize model accuracy.
- **Extension to Multi-Class Classification:** Adapting the model to classify reviews into multiple categories beyond positive and negative sentiments.

## Limitations

- **Computational Resources:** The adaptation layers, LoRA and RoSA, were implemented under limited computational resources. This restriction not only affected the training duration but potentially impacted the model's ability to generalize over a larger dataset.
- **Dataset Size:** Only a fraction (10%) of the available dataset was used in this project as training set. This smaller subset might not fully represent the complexity and variability of the entire dataset, which can lead to lower performance metrics.
- **Adaptation Layers:** The application of both LoRA and RoSA adaptation layers to only certain layers of the model might not fully leverage their potential improvements. Specifically, the performance benefits of these layers can vary based on their configurations and the dataset used. This project only applies these layers selectively, which might limit their impact on the model's overall effectiveness.

---

## Model Performance

*(This section is reserved for future updates on model performance and accuracy results.)*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
