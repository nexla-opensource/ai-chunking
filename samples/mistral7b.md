# Mistral 7B

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed

Image /page/0/Picture/2 description: The image shows the word "MistralAI" in a stylized, three-dimensional font. The letters are colored in a gradient, transitioning from yellow at the top to orange at the bottom. The word is slanted slightly upwards from left to right, and a dark brown shadow extends from the bottom and right of the letters, creating a 3D effect.

# Abstract

We introduce Mistral 7B, a 7–billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms the best open 13B model (Llama 2) across all evaluated benchmarks, and the best released 34B model (Llama 1) in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B – Instruct, that surpasses Llama 2 13B – chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license. Code: <https://github.com/mistralai/mistral-src> Webpage: <https://mistral.ai/news/announcing-mistral-7b/>

# 1 Introduction

In the rapidly evolving domain of Natural Language Processing (NLP), the race towards higher model performance often necessitates an escalation in model size. However, this scaling tends to increase computational costs and inference latency, thereby raising barriers to deployment in practical, real-world scenarios. In this context, the search for balanced models delivering both high-level performance and efficiency becomes critically essential. Our model, Mistral 7B, demonstrates that a carefully designed language model can deliver high performance while maintaining an efficient inference. Mistral 7B outperforms the previous best 13B model (Llama 2, [\[26\]](#page-8-0)) across all tested benchmarks, and surpasses the best 34B model (LLaMa 34B, [\[25\]](#page-8-1)) in mathematics and code generation. Furthermore, Mistral 7B approaches the coding performance of Code-Llama 7B [\[20\]](#page-8-2), without sacrificing performance on non-code related benchmarks.

Mistral 7B leverages grouped-query attention (GQA) [\[1\]](#page-7-0), and sliding window attention (SWA) [\[6,](#page-7-1) [3\]](#page-7-2). GQA significantly accelerates the inference speed, and also reduces the memory requirement during decoding, allowing for higher batch sizes hence higher throughput, a crucial factor for real-time applications. In addition, SWA is designed to handle longer sequences more effectively at a reduced computational cost, thereby alleviating a common limitation in LLMs. These attention mechanisms collectively contribute to the enhanced performance and efficiency of Mistral 7B.

Mistral 7B is released under the Apache 2.0 license. This release is accompanied by a reference implementation[1](#page-1-0) facilitating easy deployment either locally or on cloud platforms such as AWS, GCP, or Azure using the vLLM [\[17\]](#page-8-3) inference server and SkyPilot [2](#page-1-1). Integration with Hugging Face [3](#page-1-2) is also streamlined for easier integration. Moreover, Mistral 7B is crafted for ease of fine-tuning across a myriad of tasks. As a demonstration of its adaptability and superior performance, we present a chat model fine-tuned from Mistral 7B that significantly outperforms the Llama 2 13B – Chat model.

Mistral 7B takes a significant step in balancing the goals of getting high performance while keeping large language models efficient. Through our work, our aim is to help the community create more affordable, efficient, and high-performing language models that can be used in a wide range of real-world applications.

# 2 Architectural details

<span id="page-1-4"></span>Image /page/1/Figure/3 description: The image shows two matrices and a diagram illustrating a concept related to natural language processing or computational linguistics. The matrices on the left and in the middle represent a form of attention or relationship between words in the sentence "The cat sat on the". Each row and column corresponds to a word in the sentence, and the cells contain either a 1 or a 0, indicating the presence or absence of a relationship between the words. The lower triangle of each matrix is filled with 1s, while the upper triangle is filled with 0s, with a diagonal line separating the two regions. The matrix on the left has 1s on the lower triangle and 0s on the upper triangle, while the matrix in the middle has 1s on the lower triangle and 0s on the upper triangle, but the 0s and 1s are swapped in the upper triangle. The diagram on the right shows a series of stacked layers, each representing a different level of abstraction or processing. The bottom layer consists of a series of tokens, which are likely the individual words in the sentence. Each subsequent layer is derived from the layer below it, with a "window size" indicating the number of tokens that are considered at each step. The layers are colored in shades of red and orange, with the bottom layer being the darkest and the top layer being the lightest. The diagram illustrates how the model processes the input tokens through multiple layers to extract higher-level features or relationships.

**Vanilla Attention**

**ding Window Attention**

Figure 1: Sliding Window Attention. The number of operations in vanilla attention is quadratic in the sequence length, and the memory increases linearly with the number of tokens. At inference time, this incurs higher latency and smaller throughput due to reduced cache availability. To alleviate this issue, we use sliding window attention: each token can attend to at most  $W$  tokens from the previous layer (here,  $W = 3$ ). Note that tokens outside the sliding window still influence next word prediction. At each attention layer, information can move forward by  $W$  tokens. Hence, after  $k$  attention layers, information can move forward by up to  $k \times W$  tokens.

Mistral 7B is based on a transformer architecture [\[27\]](#page-8-4). The main parameters of the architecture are summarized in Table [1.](#page-1-3) Compared to Llama, it introduces a few changes that we summarize below.

Sliding Window Attention. SWA exploits the stacked layers of a transformer to attend information beyond the window size  $W$ . The hidden state in position  $i$  of the layer  $k$ , *h*<sup>i</sup>, attends to all hidden states from the previous layer with positions between  $i - W$  and  $i$ . Recursively, *h*<sup>i</sup> can access tokens from the input layer at a distance of up to  $W \times k$  tokens, as illustrated in Figure [1.](#page-1-4) At the last layer, using a window size of  $W = 4096$ , we have a theoretical attention span of approximately 131K tokens. In practice, for a sequence length of 16K and  $W = 4096$ , changes made to FlashAttention [\[11\]](#page-7-3) and xFormers [\[18\]](#page-8-5) yield a 2x speed improvement over a vanilla attention baseline.

<span id="page-1-3"></span>

| Parameter   | Value |
|-------------|-------|
| dim         | 4096  |
| n_layers    | 32    |
| head_dim    | 128   |
| hidden_dim  | 14336 |
| n_heads     | 32    |
| n_kv_heads  | 8     |
| window_size | 4096  |
| context_len | 8192  |
| vocab_size  | 32000 |

Table 1: Model architecture.

Rolling Buffer Cache. A fixed attention span means that we can limit our cache size using a rolling buffer cache. The cache has a fixed size of  $W$ , and the keys and values for the timestep  $i$  are stored in position  $i$  mod  $W$  of the cache. As a result, when the position  $i$  is larger than  $W$ , past values in the cache are overwritten, and the size of the cache stops increasing. We provide an illustration in Figure [2](#page-2-0) for  $W = 3$ . On a sequence length of 32k tokens, this reduces the cache memory usage by 8x, without impacting the model quality.

<span id="page-1-0"></span><sup>1</sup><https://github.com/mistralai/mistral-src>

<span id="page-1-1"></span><sup>2</sup><https://github.com/skypilot-org/skypilot>

<span id="page-1-2"></span><sup>3</sup><https://huggingface.co/mistralai>

<span id="page-2-0"></span>Image /page/2/Figure/0 description: The image shows three tables representing different timesteps (i, i+1, i+2) in a sequence prediction task. Each table has three rows, each corresponding to a different sentence. The first row represents the sentence "This is an example of ...", the second row represents "Mistral is a good ...", and the third row represents "The cat sat on the mat ...". Each cell in the table contains a word from the corresponding sentence. The cells are colored either yellow or orange, with orange indicating the word being predicted at that timestep. At timestep i, the predicted words are "is", "an" for the first sentence, "is" for the second sentence, and "sat", "on" for the third sentence. At timestep i+1, the predicted words are "an", "example" for the first sentence, "a" for the second sentence, and "sat", "on" for the third sentence. At timestep i+2, the predicted words are "of" for the first sentence, "good" for the second sentence, and "mat" for the third sentence.

Figure 2: Rolling buffer cache. The cache has a fixed size of  $W = 4$ . Keys and values for position  $i$  are stored in position i mod W of the cache. When the position i is larger than W, past values in the cache are overwritten. The hidden state corresponding to the latest generated tokens are colored in orange.

Pre-fill and Chunking. When generating a sequence, we need to predict tokens one-by-one, as each token is conditioned on the previous ones. However, the prompt is known in advance, and we can pre-fill the (k, v) cache with the prompt. If the prompt is very large, we can chunk it into smaller pieces, and pre-fill the cache with each chunk. For this purpose, we can select the window size as our chunk size. For each chunk, we thus need to compute the attention over the cache and over the chunk. Figure [3](#page-2-1) shows how the attention mask works over both the cache and the chunk.

<span id="page-2-1"></span>Image /page/2/Figure/3 description: The image shows a matrix with the words "The cat sat on the mat and saw the dog go to" along the top and the words "the dog go to" along the left side. The matrix is divided into three sections labeled "Past", "Cache", and "Current". The matrix contains 0s and 1s. The "Past" section contains all 0s. The "Cache" section contains 0s in the lower left triangle and 1s in the upper right triangle. The "Current" section contains 1s in the lower left triangle and 0s in the upper right triangle.

Figure 3: Pre-fill and chunking. During pre-fill of the cache, long sequences are chunked to limit memory usage. We process a sequence in three chunks, "The cat sat on", "the mat and saw", "the dog go to". The figure shows what happens for the third chunk ("the dog go to"): it attends itself using a causal mask (rightmost block), attends the cache using a sliding window (center block), and does not attend to past tokens as they are outside of the sliding window (left block).

# 3 Results

We compare Mistral 7B to Llama, and re-run all benchmarks with our own evaluation pipeline for fair comparison. We measure performance on a wide variety of tasks categorized as follow:

- Commonsense Reasoning (0-shot): Hellaswag [\[28\]](#page-8-6), Winogrande [\[21\]](#page-8-7), PIQA [\[4\]](#page-7-4), SIQA [\[22\]](#page-8-8), OpenbookQA [\[19\]](#page-8-9), ARC-Easy, ARC-Challenge [\[9\]](#page-7-5), CommonsenseQA [\[24\]](#page-8-10)
- World Knowledge (5-shot): NaturalQuestions [\[16\]](#page-7-6), TriviaQA [\[15\]](#page-7-7)
- Reading Comprehension (0-shot): BoolQ [\[8\]](#page-7-8), QuAC [\[7\]](#page-7-9)
- Math: GSM8K [\[10\]](#page-7-10) (8-shot) with maj@8 and MATH [\[13\]](#page-7-11) (4-shot) with maj@4
- Code: Humaneval [\[5\]](#page-7-12) (0-shot) and MBPP [\[2\]](#page-7-13) (3-shot)
- Popular aggregated results: MMLU [\[12\]](#page-7-14) (5-shot), BBH [\[23\]](#page-8-11) (3-shot), and AGI Eval [\[29\]](#page-8-12) (3-5-shot, English multiple-choice questions only)

Detailed results for Mistral 7B, Llama 2 7B/13B, and Code-Llama 7B are reported in Table [2.](#page-3-0) Figure [4](#page-3-1) compares the performance of Mistral 7B with Llama 2 7B/13B, and Llama 1 34B<sup>4</sup> in different categories. Mistral 7B surpasses Llama 2 13B across all metrics, and outperforms Llama 1 34B on most benchmarks. In particular, Mistral 7B displays a superior performance in code, mathematics, and reasoning benchmarks.

<span id="page-2-2"></span><sup>4</sup>Since Llama 2 34B was not open-sourced, we report results for Llama 1 34B.

<span id="page-3-1"></span>Image /page/3/Figure/0 description: The image contains two bar charts comparing the accuracy of different language models on various tasks. The first chart, on the left, shows the accuracy (%) of Mistral 7B, LLaMA 2 7B, LLaMA 2 13B, and LLaMA 1 34B on MMLU, Knowledge, Reasoning, and Comprehension tasks. The accuracy ranges from 30% to 70%. The second chart, on the right, shows the accuracy (%) of the same models on AGI Eval, Math, BBH, and Code tasks. The accuracy ranges from 10% to 50%. The legend indicates that Mistral 7B is represented by an orange bar, LLaMA 2 7B by a red bar, LLaMA 2 13B by a blue bar, and LLaMA 1 34B by a yellow bar.

Figure 4: Performance of Mistral 7B and different Llama models on a wide range of benchmarks. All models were re-evaluated on all metrics with our evaluation pipeline for accurate comparison. Mistral 7B significantly outperforms Llama 2 7B and Llama 2 13B on all benchmarks. It is also vastly superior to Llama 1 34B in mathematics, code generation, and reasoning benchmarks.

<span id="page-3-0"></span>

| Model         | Modality   | MMLU         | HellaSwag    | WinoG | PIQA         | Arc-e        | Arc-c        | NQ           | TriviaQA     | HumanEval    | MBPP         | MATH         | GSM8E        |
|---------------|------------|--------------|--------------|-------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| LLaMA 2 7B    | Pretrained | 44.4%        | 77.1%        | 69.5% | 77.9%        | 68.7%        | 43.2%        | 24.7%        | 63.8%        | 11.6%        | 26.1%        | 3.9%         | 16.0%        |
| LLaMA 2 13B   | Pretrained | 55.6%        | <b>80.7%</b> | 72.9% | 80.8%        | 75.2%        | 48.8%        | <b>29.0%</b> | <b>69.6%</b> | 18.9%        | 35.4%        | 6.0%         | 34.3%        |
| Code-Llama 7B | Finetuned  | 36.9%        | 62.9%        | 62.3% | 72.8%        | 59.4%        | 34.5%        | 11.0%        | 34.9%        | <b>31.1%</b> | <b>52.5%</b> | 5.2%         | 20.8%        |
| Mistral 7B    | Pretrained | <b>60.1%</b> | <b>81.3%</b> | 75.3% | <b>83.0%</b> | <b>80.0%</b> | <b>55.5%</b> | 28.8%        | 69.9%        | <b>30.5%</b> | 47.5%        | <b>13.1%</b> | <b>52.2%</b> |

Table 2: Comparison of Mistral 7B with Llama. Mistral 7B outperforms Llama 2 13B on all metrics, and approaches the code performance of Code-Llama 7B without sacrificing performance on non-code benchmarks.

Size and Efficiency. We computed "equivalent model sizes" of the Llama 2 family, aiming to understand Mistral 7B models' efficiency in the cost-performance spectrum (see Figure [5\)](#page-4-0). When evaluated on reasoning, comprehension, and STEM reasoning (specifically MMLU), Mistral 7B mirrored performance that one might expect from a Llama 2 model with more than 3x its size. On the Knowledge benchmarks, Mistral 7B's performance achieves a lower compression rate of 1.9x, which is likely due to its limited parameter count that restricts the amount of knowledge it can store.

Evaluation Differences. On some benchmarks, there are some differences between our evaluation protocol and the one reported in the Llama 2 paper: 1) on MBPP, we use the hand-verified subset 2) on TriviaQA, we do not provide Wikipedia contexts.

# 4 Instruction Finetuning

To evaluate the generalization capabilities of Mistral 7B, we fine-tuned it on instruction datasets publicly available on the Hugging Face repository. No proprietary data or training tricks were utilized: Mistral 7B - Instruct model is a simple and preliminary demonstration that the base model can easily be fine-tuned to achieve good performance. In Table [3,](#page-3-2) we observe that the resulting model, Mistral 7B - Instruct, exhibits superior performance compared to all 7B models on MT-Bench, and is comparable to 13B – Chat models. An independent human evaluation was conducted on <https://llmboxing.com/leaderboard>.

<span id="page-3-2"></span>

| Model               | Chatbot Arena<br>ELO Rating | MT Bench      |
|---------------------|-----------------------------|---------------|
| WizardLM 13B v1.2   | 1047                        | 7.2           |
| Mistral 7B Instruct | 1031                        | 6.84 +/- 0.07 |
| Llama 2 13B Chat    | 1012                        | 6.65          |
| Vicuna 13B          | 1041                        | 6.57          |
| Llama 2 7B Chat     | 985                         | 6.27          |
| Vicuna 7B           | 997                         | 6.17          |
| Alpaca 13B          | 914                         | 4.53          |

Table 3: Comparison of Chat models. Mistral 7B -Instruct outperforms all 7B models on MT-Bench, and is comparable to 13B - Chat models.

In this evaluation, participants were provided with a set of questions along with anonymous responses from two models and were asked to select their preferred response, as illustrated in Figure [6.](#page-6-0) As of October 6, 2023, the outputs generated by Mistral 7B were preferred 5020 times, compared to 4143 times for Llama 2 13B.

<span id="page-4-0"></span>Image /page/4/Figure/0 description: The image contains four line plots comparing the performance of LLaMA 2 and Mistral models on different tasks. The plots show the relationship between model size (in billion parameters) and performance (in percentage) for MMLU, Reasoning, Knowledge, and Comprehension. In each plot, LLaMA 2 is represented by a red line with circular markers, while Mistral is represented by an orange square marker. The x-axis represents the model size, with values of 7, 13, 34, and 70 billion parameters. The y-axis represents the performance percentage. For MMLU, the LLaMA 2 line shows an increasing trend with model size, starting at around 45% for 7B parameters and reaching around 70% for 70B parameters. The Mistral model achieves around 60% performance. An annotation indicates that the Mistral model's performance is equivalent to an effective LLaMA size of 23B (3.3x). For Reasoning, the LLaMA 2 line also shows an increasing trend, starting at around 63% for 7B parameters and reaching around 71% for 70B parameters. The Mistral model achieves around 69% performance. An annotation indicates that the Mistral model's performance is equivalent to an effective LLaMA size of 38B (5.4x). For Knowledge, the LLaMA 2 line shows an increasing trend, starting at around 44% for 7B parameters and reaching around 71% for 70B parameters. The Mistral model achieves around 49% performance. An annotation indicates that the Mistral model's performance is equivalent to an effective LLaMA size of 13B (1.9x). For Comprehension, the LLaMA 2 line shows an increasing trend, starting at around 59% for 7B parameters and reaching around 68% for 70B parameters. The Mistral model achieves around 64% performance. An annotation indicates that the Mistral model's performance is equivalent to an effective LLaMA size of 21B (3x).

Figure 5: Results on MMLU, commonsense reasoning, world knowledge and reading comprehension for Mistral 7B and Llama 2 (7B/13B/70B). Mistral 7B largely outperforms Llama 2 13B on all evaluations, except on knowledge benchmarks, where it is on par (this is likely due to its limited parameter count, which limits the amount of knowledge it can compress).

# 5 Adding guardrails for front-facing applications

The ability to enforce guardrails when it comes to AI generation is important for front-facing applications. In this section, we highlight how to leverage system prompting to optionally enforce output constraints on top of our models. Additionally, we showcase the ability of Mistral 7B to perform fine-grained content moderation, which can be useful to enforce quality content in applications.

### 5.1 System prompt to enforce guardrails

We introduce a system prompt (see below) to guide the model to generate answers within specified guardrails, similar to the work done with Llama 2. Using this prompt allows the user to move on the Pareto front of model utility / guardrails enforcement, as indicated in Table [4.](#page-4-1)

Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.

We use a set of 175 unsafe prompts for evaluating safety. With the recommended system prompt, the model properly declines to answer 100% of the harmful questions.

As an illustration, we provide in Table [5](#page-5-0) the answers of both Mistral 7B - Instruct and Llama 2 Chat 13B to the question *How to kill a linux process* with system prompts activated. We observe that Mistral 7B provides a correct response while Llama 2 declines to answer. Note that on this specific question, both models answer correctly when system prompts are deactivated.

<span id="page-4-1"></span>

| Guardrails            | MT Bench    |
|-----------------------|-------------|
| No system prompt      | 6.84 ± 0.07 |
| Llama 2 system prompt | 6.38 ± 0.07 |
| Mistral system prompt | 6.58 ± 0.05 |

Table 4: System prompts. Mean official MT Bench score over 10 iterations with standard deviation for Mistral 7B - Instruct. For reference, Llama 2 13B – Chat reports official results of 6.65.

### 5.2 Content moderation with self-reflection

Mistral 7B - Instruct can be used as a content moderator: the model itself is able to accurately classify a user prompt or its generated answer as being either acceptable or falling into one of the following categories: Illegal activities such as terrorism, child abuse or fraud; Hateful, harassing or violent content such as discrimination, self-harm or bullying; Unqualified advice for instance in legal, medical or financial domains.

<span id="page-5-0"></span>

| Model                                               | Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mistral 7B – Instruct<br>with Mistral system prompt | To kill a Linux process, you can use the `kill` command followed by the process ID (PID)<br>of the process you want to terminate. For example, to kill process with PID 1234, you<br>would run the command `kill 1234`. It's important to note that killing a process can<br>have unintended consequences, so it's generally a good idea to only kill processes that<br>you are certain you want to terminate. Additionally, it's generally a good idea to use the<br>`kill` command with caution, as it can be used to terminate critical system processes that<br>are necessary for the operation of the system. |
| Llama 2 13B – Chat<br>with Llama 2 system prompt    | As a helpful and respectful assistant, I cannot provide information on how to kill a Linux<br>process. It is not appropriate or ethical to intentionally harm or terminate processes, as it<br>can cause damage to the system and potentially result in data loss. Additionally, it is<br>important to respect the autonomy and security of other users' processes, as they may be<br>critical to their work or personal activities. Instead, I suggest that you explore other<br>options to address any issues or concerns you may have.                                                                          |

Table 5: Comparison between Mistral and Llama system prompts. Mistral provides the right answer whereas Llama 2 declines to answer to the question *How to kill a linux process*.

To do so, we designed a self-reflection prompt that makes Mistral 7B classify a prompt or a generated answer. We evaluated self-reflection on our manually curated and balanced dataset of adversarial and standard prompts and got a precision of 99.4% for a recall of 95.6% (considering acceptable prompts as positives).

The use cases are vast, from moderating comments on social media or forums to brand monitoring on the internet. In particular, the end user is able to select afterwards which categories to effectively filter based on their particular use-case.

# 6 Conclusion

Our work on Mistral 7B demonstrates that language models may compress knowledge more than what was previously thought. This opens up interesting perspectives: the field has so far put the emphasis on scaling laws in 2 dimensions (directly associating model capabilities to training cost, as in [\[14\]](#page-7-15)); the problem is rather 3 dimensional (model capabilities, training cost, inference cost), and much remains to be explored to obtain the best performance with the smallest possible model.

# Acknowledgements

We are grateful to CoreWeave for their 24/7 help in marshalling our cluster. We thank the CINECA/EuroHPC team, and in particular the operators of Leonardo, for their resources and help. We thank the maintainers of FlashAttention, vLLM, xFormers, Skypilot for their precious assistance in implementing new features and integrating their solutions into ours. A huge thanks to Tri Dao and Daniel Haziza for helping include Mistral related changes to FlashAttention and xFormers on a tight schedule. We thank the teams of Hugging Face, AWS, GCP, Azure ML for their intense help in making our model compatible everywhere.

<span id="page-6-0"></span>

| Llama 2 13b  | 1 Health Remaining |
|--------------|--------------------|
| M Mistral 7b | 5 Health Remaining |

Can you recommend a foundational and accessible book about quantum physics?

| Copy Link to Question |
|-----------------------|
| Next Round (Enter) →  |

Round Winner: mistral-7b-instruct-v0.1

Image /page/6/Picture/4 description: A cartoon image shows an orange letter M wearing red boxing gloves. The letter M has two black dots for eyes and is set against a dark background.

A great book for a foundational
understanding of quantum physics is
"The Quantum Universe: Everything
that Can Happen Does Happen" by
Brian Cox and Jeff Forshaw. This
book is accessible and easy to
understand, making it a great
choice for beginners. It covers all
the basics of quantum mechanics,
including superposition,
entanglement, and the uncertainty
principle. The authors also explore
the implications of quantum physics
for our understanding of the
universe, including the nature of
time and space. Overall, "The
Quantum Universe" is an excellent
choice for anyone looking to gain a
foundational understanding of this
fascinating field

Certainly! For a foundational and
accessible introduction to quantum
physics, I highly recommend "The
Feynman Lectures on Physics" by
Richard Feynman. This classic text
provides a clear and engaging
overview of the principles and
concepts of quantum mechanics, with
minimal mathematical formalism and
plenty of illustrations and
examples. It's a great starting
point for anyone looking to gain a
deeper understanding of quantum
physics without getting bogged down
in too much technical detail.

Figure 6: Human evaluation of Mistral 7B - Instruct vs Llama 2 13B - Chat Example. An example of human evaluation from <llmboxing.com>. The question asks for recommendations of books in quantum physics. Llama 2 13B - Chat recommends a general physics book, while Mistral 7B - Instruct recommends a more relevant book on quantum physics and describes in the contents in more detail.

# References

- <span id="page-7-0"></span>[1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*, 2023.
- <span id="page-7-13"></span>[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.
- <span id="page-7-2"></span>[3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020.
- <span id="page-7-4"></span>[4] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, 2020.
- <span id="page-7-12"></span>[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.
- <span id="page-7-1"></span>[6] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*, 2019.
- <span id="page-7-9"></span>[7] Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. Quac: Question answering in context. *arXiv preprint arXiv:1808.07036*, 2018.
- <span id="page-7-8"></span>[8] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. *arXiv preprint arXiv:1905.10044*, 2019.
- <span id="page-7-5"></span>[9] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.
- <span id="page-7-10"></span>[10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.
- <span id="page-7-3"></span>[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems*, 2022.
- <span id="page-7-14"></span>[12] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. *arXiv preprint arXiv:2009.03300*, 2020.
- <span id="page-7-11"></span>[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.
- <span id="page-7-15"></span>[14] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karén Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre. An empirical analysis of compute-optimal large language model training. In *Advances in Neural Information Processing Systems*, volume 35, 2022.
- <span id="page-7-7"></span>[15] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. *arXiv preprint arXiv:1705.03551*, 2017.
- <span id="page-7-6"></span>[16] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7:453–466, 2019.
- <span id="page-8-3"></span>[17] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
- <span id="page-8-5"></span>[18] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. [https://github.com/](https://github.com/facebookresearch/xformers) [facebookresearch/xformers](https://github.com/facebookresearch/xformers), 2022.
- <span id="page-8-9"></span>[19] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. *arXiv preprint arXiv:1809.02789*, 2018.
- <span id="page-8-2"></span>[20] Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. *arXiv preprint arXiv:2308.12950*, 2023.
- <span id="page-8-7"></span>[21] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.
- <span id="page-8-8"></span>[22] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. *arXiv preprint arXiv:1904.09728*, 2019.
- <span id="page-8-11"></span>[23] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. *arXiv preprint arXiv:2210.09261*, 2022.
- <span id="page-8-10"></span>[24] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. *arXiv preprint arXiv:1811.00937*, 2018.
- <span id="page-8-1"></span>[25] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.
- <span id="page-8-0"></span>[26] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-8-4"></span>[27] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-8-6"></span>[28] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? *arXiv preprint arXiv:1905.07830*, 2019.
- <span id="page-8-12"></span>[29] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. *arXiv preprint arXiv:2304.06364*, 2023.