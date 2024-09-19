# LLM Training - Data Preparation

**这些是我从非常推荐的书籍** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **中整理的笔记，附加了一些额外信息。**

## Basic Information

您应该先阅读这篇文章，以了解一些基本概念：

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
这个初始阶段的目标非常简单：**以某种合理的方式将输入划分为标记（ID）**。
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
这个第二阶段的目标非常简单：**对输入数据进行采样，并为训练阶段准备数据，通常通过将数据集分隔为特定长度的句子，并生成预期的响应。**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
这个第三阶段的目标非常简单：**为词汇表中的每个标记分配一个所需维度的向量以训练模型。** 词汇表中的每个单词将在X维空间中有一个点。\
请注意，最初每个单词在空间中的位置是“随机”初始化的，这些位置是可训练的参数（在训练过程中会得到改善）。

此外，在标记嵌入期间**创建了另一层嵌入**，它表示（在这种情况下）**单词在训练句子中的绝对位置**。这样，句子中不同位置的单词将具有不同的表示（含义）。
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
这个第四阶段的目标非常简单：**应用一些注意机制**。这些将是许多**重复的层**，将**捕捉词汇表中单词与当前用于训练LLM的句子中其邻居的关系**。\
为此使用了许多层，因此将有许多可训练的参数来捕捉这些信息。
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
这个第五阶段的目标非常简单：**开发完整LLM的架构**。将所有内容整合在一起，应用所有层，并创建所有函数以生成文本或将文本转换为ID及其反向操作。

该架构将用于训练和预测文本。
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
这个第六阶段的目标非常简单：**从头开始训练模型**。为此，将使用之前的LLM架构，通过对数据集进行循环，使用定义的损失函数和优化器来训练模型的所有参数。
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
使用**LoRA大大减少了**所需的**微调**已训练模型的计算量。
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
本节的目标是展示如何微调一个已经预训练的模型，以便LLM选择给定文本被分类到每个给定类别的**概率**（例如，文本是否为垃圾邮件）。
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
本节的目标是展示如何**微调一个已经预训练的模型以遵循指令**，而不仅仅是生成文本，例如，作为聊天机器人响应任务。
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
