# LLM Training - Datenvorbereitung

**Dies sind meine Notizen aus dem sehr empfohlenen Buch** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **mit einigen zusätzlichen Informationen.**

## Grundinformationen

Sie sollten mit dem Lesen dieses Beitrags beginnen, um einige grundlegende Konzepte zu verstehen, die Sie wissen sollten:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenisierung

{% hint style="success" %}
Das Ziel dieser ersten Phase ist sehr einfach: **Teilen Sie die Eingabe in Tokens (IDs) auf eine Weise, die Sinn macht**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Datenauswahl

{% hint style="success" %}
Das Ziel dieser zweiten Phase ist sehr einfach: **Wählen Sie die Eingabedaten aus und bereiten Sie sie für die Trainingsphase vor, indem Sie den Datensatz normalerweise in Sätze einer bestimmten Länge unterteilen und auch die erwartete Antwort generieren.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token-Embeddings

{% hint style="success" %}
Das Ziel dieser dritten Phase ist sehr einfach: **Weisen Sie jedem der vorherigen Tokens im Vokabular einen Vektor der gewünschten Dimensionen zu, um das Modell zu trainieren.** Jedes Wort im Vokabular wird einen Punkt in einem Raum von X Dimensionen haben.\
Beachten Sie, dass die Position jedes Wortes im Raum zunächst "zufällig" initialisiert wird und diese Positionen trainierbare Parameter sind (während des Trainings verbessert werden).

Darüber hinaus wird während des Token-Embeddings **eine weitere Schicht von Embeddings erstellt**, die (in diesem Fall) die **absolute Position des Wortes im Trainingssatz** darstellt. Auf diese Weise hat ein Wort an verschiedenen Positionen im Satz eine unterschiedliche Darstellung (Bedeutung).
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Aufmerksamkeitsmechanismen

{% hint style="success" %}
Das Ziel dieser vierten Phase ist sehr einfach: **Wenden Sie einige Aufmerksamkeitsmechanismen an**. Dies werden viele **wiederholte Schichten** sein, die die **Beziehung eines Wortes im Vokabular zu seinen Nachbarn im aktuellen Satz, der zum Trainieren des LLM verwendet wird, erfassen**.\
Es werden viele Schichten dafür verwendet, sodass viele trainierbare Parameter diese Informationen erfassen werden.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM-Architektur

{% hint style="success" %}
Das Ziel dieser fünften Phase ist sehr einfach: **Entwickeln Sie die Architektur des gesamten LLM**. Fügen Sie alles zusammen, wenden Sie alle Schichten an und erstellen Sie alle Funktionen, um Text zu generieren oder Text in IDs und umgekehrt zu transformieren.

Diese Architektur wird sowohl für das Training als auch für die Vorhersage von Text nach dem Training verwendet.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Vortraining & Laden von Modellen

{% hint style="success" %}
Das Ziel dieser sechsten Phase ist sehr einfach: **Trainieren Sie das Modell von Grund auf neu**. Dazu wird die vorherige LLM-Architektur mit einigen Schleifen über die Datensätze verwendet, wobei die definierten Verlustfunktionen und der Optimierer verwendet werden, um alle Parameter des Modells zu trainieren.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA-Verbesserungen beim Feintuning

{% hint style="success" %}
Die Verwendung von **LoRA reduziert die benötigte Berechnung** erheblich, um **bereits trainierte Modelle fein abzustimmen**.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Feintuning für Klassifikation

{% hint style="success" %}
Das Ziel dieses Abschnitts ist zu zeigen, wie man ein bereits vortrainiertes Modell fein abstimmt, sodass das LLM anstelle von neuem Text die **Wahrscheinlichkeiten des gegebenen Textes für jede der gegebenen Kategorien** (wie ob ein Text Spam ist oder nicht) angibt.
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Feintuning zur Befolgung von Anweisungen

{% hint style="success" %}
Das Ziel dieses Abschnitts ist zu zeigen, wie man **ein bereits vortrainiertes Modell fein abstimmt, um Anweisungen zu befolgen**, anstatt nur Text zu generieren, zum Beispiel, um auf Aufgaben als Chatbot zu antworten.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
