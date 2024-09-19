# LLM Training - Data Preparation

**Estas são minhas anotações do livro muito recomendado** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **com algumas informações extras.**

## Basic Information

Você deve começar lendo este post para alguns conceitos básicos que você deve conhecer:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
O objetivo desta fase inicial é muito simples: **Dividir a entrada em tokens (ids) de uma maneira que faça sentido**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
O objetivo desta segunda fase é muito simples: **Amostrar os dados de entrada e prepará-los para a fase de treinamento, geralmente separando o conjunto de dados em frases de um comprimento específico e gerando também a resposta esperada.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
O objetivo desta terceira fase é muito simples: **Atribuir a cada um dos tokens anteriores no vocabulário um vetor das dimensões desejadas para treinar o modelo.** Cada palavra no vocabulário será um ponto em um espaço de X dimensões.\
Note que inicialmente a posição de cada palavra no espaço é apenas inicializada "aleatoriamente" e essas posições são parâmetros treináveis (serão melhoradas durante o treinamento).

Além disso, durante a incorporação de tokens **outra camada de incorporações é criada** que representa (neste caso) a **posição absoluta da palavra na frase de treinamento**. Dessa forma, uma palavra em diferentes posições na frase terá uma representação (significado) diferente.
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
O objetivo desta quarta fase é muito simples: **Aplicar alguns mecanismos de atenção**. Estes serão muitas **camadas repetidas** que vão **capturar a relação de uma palavra no vocabulário com seus vizinhos na frase atual sendo usada para treinar o LLM**.\
Muitas camadas são usadas para isso, então muitos parâmetros treináveis estarão capturando essa informação.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
O objetivo desta quinta fase é muito simples: **Desenvolver a arquitetura do LLM completo**. Juntar tudo, aplicar todas as camadas e criar todas as funções para gerar texto ou transformar texto em IDs e vice-versa.

Esta arquitetura será usada tanto para treinar quanto para prever texto após ter sido treinada.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
O objetivo desta sexta fase é muito simples: **Treinar o modelo do zero**. Para isso, a arquitetura LLM anterior será usada com alguns loops sobre os conjuntos de dados usando as funções de perda e otimizador definidos para treinar todos os parâmetros do modelo.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
O uso de **LoRA reduz muito a computação** necessária para **ajustar** modelos já treinados.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
O objetivo desta seção é mostrar como ajustar um modelo já pré-treinado para que, em vez de gerar novo texto, o LLM selecione e forneça as **probabilidades do texto dado ser categorizado em cada uma das categorias dadas** (como se um texto é spam ou não).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
O objetivo desta seção é mostrar como **ajustar um modelo já pré-treinado para seguir instruções** em vez de apenas gerar texto, por exemplo, respondendo a tarefas como um chatbot.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
