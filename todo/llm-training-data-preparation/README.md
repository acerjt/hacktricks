# LLM Training - Data Preparation

**Queste sono le mie note dal libro molto raccomandato** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **con alcune informazioni extra.**

## Basic Information

Dovresti iniziare leggendo questo post per alcuni concetti di base che dovresti conoscere:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
L'obiettivo di questa fase iniziale è molto semplice: **Dividere l'input in token (ids) in un modo che abbia senso**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
L'obiettivo di questa seconda fase è molto semplice: **Campionare i dati di input e prepararli per la fase di addestramento solitamente separando il dataset in frasi di una lunghezza specifica e generando anche la risposta attesa.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
L'obiettivo di questa terza fase è molto semplice: **Assegnare a ciascuno dei token precedenti nel vocabolario un vettore delle dimensioni desiderate per addestrare il modello.** Ogni parola nel vocabolario sarà un punto in uno spazio di X dimensioni.\
Nota che inizialmente la posizione di ciascuna parola nello spazio è semplicemente inizializzata "randomicamente" e queste posizioni sono parametri addestrabili (saranno migliorati durante l'addestramento).

Inoltre, durante l'embedding dei token **viene creata un'altra layer di embeddings** che rappresenta (in questo caso) la **posizione assoluta della parola nella frase di addestramento**. In questo modo, una parola in posizioni diverse nella frase avrà una rappresentazione (significato) diversa.
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
L'obiettivo di questa quarta fase è molto semplice: **Applicare alcuni meccanismi di attenzione**. Questi saranno molti **layer ripetuti** che andranno a **catturare la relazione di una parola nel vocabolario con i suoi vicini nella frase attuale utilizzata per addestrare il LLM**.\
Vengono utilizzati molti layer per questo, quindi molti parametri addestrabili andranno a catturare queste informazioni.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
L'obiettivo di questa quinta fase è molto semplice: **Sviluppare l'architettura del LLM completo**. Mettere tutto insieme, applicare tutti i layer e creare tutte le funzioni per generare testo o trasformare testo in ID e viceversa.

Questa architettura sarà utilizzata sia per l'addestramento che per la previsione del testo dopo che è stato addestrato.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
L'obiettivo di questa sesta fase è molto semplice: **Addestrare il modello da zero**. Per questo verrà utilizzata l'architettura LLM precedente con alcuni loop che attraversano i dataset utilizzando le funzioni di perdita e l'ottimizzatore definiti per addestrare tutti i parametri del modello.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
L'uso di **LoRA riduce notevolmente il calcolo** necessario per **fine-tune** modelli già addestrati.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
L'obiettivo di questa sezione è mostrare come fare fine-tuning a un modello già pre-addestrato in modo che, invece di generare nuovo testo, il LLM selezioni e fornisca le **probabilità che il testo fornito venga categorizzato in ciascuna delle categorie date** (come se un testo è spam o meno).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
L'obiettivo di questa sezione è mostrare come **fine-tune un modello già pre-addestrato per seguire istruzioni** piuttosto che semplicemente generare testo, ad esempio, rispondere a compiti come un chatbot.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
