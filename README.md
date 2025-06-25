
# Next Utterance Prediction for Mental Health Counseling

**Team Members:**
- Abhay Shakya - abhay24108@iiitd.ac.in
- Chaitanya Lakhchaura - chaitanya24027@iiitd.ac.in
- Ayush Kumar Verma - ayush24025@iiitd.ac.in

**Institution:**
- IIIT Delhi, New Delhi, India

## Overview

This project focuses on developing an NLP-based system to predict the next appropriate utterance in mental health counseling conversations. The goal is to generate contextually relevant, empathetic, and emotionally supportive responses that enhance therapeutic dialogues. By using real-world counseling dialogue data, the system generates the next conversational turn in a manner that is both emotionally intelligent and aligned with the context of the conversation.

## Objectives

- Generate responses that are contextually relevant, empathetic, and emotionally supportive.
- Train a model on real-world counseling dialogues to improve the flow of therapeutic conversations.
- Evaluate the responses using BLEU and BERT scores to ensure their fluency and contextual alignment.

## Methodology

We use transformer-based models (BART and GPT-2) for sequence generation. Both models are fine-tuned to handle speaker differentiation and extended dialogues in therapy settings.

1. **Speaker-specific embeddings:** Differentiating between therapist (T) and patient (P) to create more accurate and contextually appropriate responses.
2. **Positional embeddings adjustment:** Adjusting the model to handle longer conversations by extending the sequence length.
3. **Text Preprocessing:** Involves tokenizing, padding, and removing unnecessary tokens to standardize the input text for training.

## Dataset

The dataset consists of therapy session conversations, labeled with speaker tags (T for therapist, P for patient). Each conversation turn is separated by a special separator token `[SEP]`. The dataset includes:
- **4008 rows** for input and target features
- **968 rows** for validation
- **576 rows** for testing

## Results

Evaluation metrics for the model:
- **BLEU Score:** 0.0162
- **BERT Precision:** 0.8583
- **BERT Recall:** 0.8470
- **BERT F1-Score:** 0.8523

These metrics show that the model effectively captures semantic relevance and emotional appropriateness in therapy dialogue.

## Future Work

- Expand the dataset to improve generalization.
- Incorporate more advanced techniques such as attention mechanisms and reinforcement learning.
- Use larger models to improve the system’s ability to handle complex therapeutic dialogues.

## References

1. **BART:** Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension - [Link](https://arxiv.org/pdf/1910.13461)
2. **Language Models are Unsupervised Multitask Learners** - [Link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. **Response-act Guided Reinforced Dialogue Generation for Mental Health Counseling** - [Link](https://arxiv.org/abs/2301.12729)
4. **Counseling Summarization using Mental Health Knowledge Guided Utterance Filtering** - [Link](https://arxiv.org/pdf/2206.03886)
