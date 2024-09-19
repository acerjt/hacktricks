# LLM Training - Data Preparation

**Estas son mis notas del libro muy recomendado** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **con algo de información extra.**

## Basic Information

Deberías comenzar leyendo esta publicación para algunos conceptos básicos que deberías conocer:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
El objetivo de esta fase inicial es muy simple: **Dividir la entrada en tokens (ids) de una manera que tenga sentido**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
El objetivo de esta segunda fase es muy simple: **Muestrear los datos de entrada y prepararlos para la fase de entrenamiento, generalmente separando el conjunto de datos en oraciones de una longitud específica y generando también la respuesta esperada.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
El objetivo de esta tercera fase es muy simple: **Asignar a cada uno de los tokens anteriores en el vocabulario un vector de las dimensiones deseadas para entrenar el modelo.** Cada palabra en el vocabulario será un punto en un espacio de X dimensiones.\
Ten en cuenta que inicialmente la posición de cada palabra en el espacio se inicializa "aleatoriamente" y estas posiciones son parámetros entrenables (se mejorarán durante el entrenamiento).

Además, durante la incrustación de tokens **se crea otra capa de incrustaciones** que representa (en este caso) la **posición absoluta de la palabra en la oración de entrenamiento**. De esta manera, una palabra en diferentes posiciones en la oración tendrá una representación (significado) diferente.
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
El objetivo de esta cuarta fase es muy simple: **Aplicar algunos mecanismos de atención**. Estos serán muchas **capas repetidas** que van a **capturar la relación de una palabra en el vocabulario con sus vecinos en la oración actual que se está utilizando para entrenar el LLM**.\
Se utilizan muchas capas para esto, por lo que muchos parámetros entrenables van a capturar esta información.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
El objetivo de esta quinta fase es muy simple: **Desarrollar la arquitectura del LLM completo**. Juntar todo, aplicar todas las capas y crear todas las funciones para generar texto o transformar texto a IDs y viceversa.

Esta arquitectura se utilizará tanto para entrenar como para predecir texto después de haber sido entrenada.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
El objetivo de esta sexta fase es muy simple: **Entrenar el modelo desde cero**. Para esto se utilizará la arquitectura LLM anterior con algunos bucles sobre los conjuntos de datos utilizando las funciones de pérdida y optimizador definidos para entrenar todos los parámetros del modelo.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
El uso de **LoRA reduce mucho la computación** necesaria para **ajustar** modelos ya entrenados.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
El objetivo de esta sección es mostrar cómo ajustar un modelo ya preentrenado para que, en lugar de generar nuevo texto, el LLM seleccione y dé las **probabilidades de que el texto dado sea categorizado en cada una de las categorías dadas** (como si un texto es spam o no).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
El objetivo de esta sección es mostrar cómo **ajustar un modelo ya preentrenado para seguir instrucciones** en lugar de solo generar texto, por ejemplo, respondiendo a tareas como un chatbot.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
