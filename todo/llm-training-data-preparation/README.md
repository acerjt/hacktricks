# LLMトレーニング - データ準備

**これは非常に推奨される本** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **からの私のメモで、いくつかの追加情報が含まれています。**

## 基本情報

知っておくべき基本概念については、この投稿を読むことから始めるべきです：

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. トークン化

{% hint style="success" %}
この初期段階の目標は非常にシンプルです：**入力を意味のある方法でトークン（ID）に分割すること**。
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. データサンプリング

{% hint style="success" %}
この第二段階の目標は非常にシンプルです：**入力データをサンプリングし、通常は特定の長さの文にデータセットを分け、期待される応答も生成することでトレーニングフェーズの準備をすること**。
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. トークン埋め込み

{% hint style="success" %}
この第三段階の目標は非常にシンプルです：**語彙内の各トークンに対して、モデルをトレーニングするために必要な次元のベクトルを割り当てること**。語彙内の各単語はX次元の空間内の点になります。\
最初は各単語の空間内の位置は「ランダムに」初期化され、これらの位置はトレーニング中に改善されるトレーニング可能なパラメータです。

さらに、トークン埋め込み中に**別の埋め込み層が作成され**、これは（この場合）**トレーニング文内の単語の絶対位置を表します**。このように、文内の異なる位置にある単語は異なる表現（意味）を持ちます。
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. アテンションメカニズム

{% hint style="success" %}
この第四段階の目標は非常にシンプルです：**いくつかのアテンションメカニズムを適用すること**。これらは**語彙内の単語と現在トレーニング中の文内の隣接単語との関係を捉えるための多くの**繰り返し層**になります。\
これには多くの層が使用されるため、多くのトレーニング可能なパラメータがこの情報を捉えることになります。
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLMアーキテクチャ

{% hint style="success" %}
この第五段階の目標は非常にシンプルです：**完全なLLMのアーキテクチャを開発すること**。すべてをまとめ、すべての層を適用し、テキストを生成したり、テキストをIDに変換したりその逆を行うためのすべての関数を作成します。

このアーキテクチャは、トレーニング後のテキストの予測にも使用されます。
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. プレトレーニングとモデルの読み込み

{% hint style="success" %}
この第六段階の目標は非常にシンプルです：**ゼロからモデルをトレーニングすること**。これには、定義された損失関数とオプティマイザを使用して、データセットをループしながらすべてのパラメータをトレーニングするために、前のLLMアーキテクチャが使用されます。
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRAによるファインチューニングの改善

{% hint style="success" %}
**LoRAの使用は、すでにトレーニングされたモデルをファインチューニングするために必要な計算を大幅に削減します**。
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. 分類のためのファインチューニング

{% hint style="success" %}
このセクションの目標は、すでにプレトレーニングされたモデルをファインチューニングする方法を示すことです。新しいテキストを生成するのではなく、LLMが**与えられたテキストが各カテゴリに分類される確率を選択すること**です（例えば、テキストがスパムかどうか）。
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. 指示に従うためのファインチューニング

{% hint style="success" %}
このセクションの目標は、**テキストを生成するだけでなく、指示に従うためにすでにプレトレーニングされたモデルをファインチューニングする方法を示すこと**です。例えば、チャットボットとしてタスクに応答することです。
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
