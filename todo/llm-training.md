# LLM Training

## Tokenizing

Tokenizing consiste à séparer les données en morceaux spécifiques et à leur attribuer des ID spécifiques (nombres).\
Un tokenizer très simple pour les textes pourrait simplement prendre chaque mot d'un texte séparément, ainsi que les symboles de ponctuation et supprimer les espaces.\
Par conséquent, `"Hello, world!"` serait : `["Hello", ",", "world", "!"]`

Ensuite, afin d'attribuer à chacun des mots et symboles un ID de token (nombre), il est nécessaire de créer le **vocabulaire** du tokenizer. Si vous tokenisez par exemple un livre, cela pourrait être **tous les mots différents du livre** dans l'ordre alphabétique avec quelques tokens supplémentaires comme :

* `[BOS] (Début de séquence)`: Placé au début d'un texte, il indique le début d'un texte (utilisé pour séparer des textes non liés).
* `[EOS] (Fin de séquence)`: Placé à la fin d'un texte, il indique la fin d'un texte (utilisé pour séparer des textes non liés).
* `[PAD] (remplissage)`: Lorsque la taille d'un lot est supérieure à un (généralement), ce token est utilisé pour augmenter la longueur de ce lot afin qu'il soit aussi grand que les autres.
* `[UNK] (inconnu)`: Pour représenter des mots inconnus.

En suivant l'exemple, ayant tokenisé un texte en attribuant à chaque mot et symbole du texte une position dans le vocabulaire, la phrase tokenisée `"Hello, world!"` -> `["Hello", ",", "world", "!"]` serait quelque chose comme : `[64, 455, 78, 467]` en supposant que `Hello` est à la position 64, "`,"` est à la position `455`... dans le tableau de vocabulaire résultant.

Cependant, si dans le texte utilisé pour générer le vocabulaire le mot `"Bye"` n'existait pas, cela donnerait : `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` en supposant que le token pour `[UNK]` est à 987.

### BPE - Byte Pair Encoding

Afin d'éviter des problèmes comme la nécessité de tokeniser tous les mots possibles pour les textes, les LLMs comme GPT utilisent BPE qui encode essentiellement **des paires de bytes fréquentes** pour réduire la taille du texte dans un format plus optimisé jusqu'à ce qu'il ne puisse plus être réduit (voir [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Notez qu'ainsi il n'y a pas de mots "inconnus" pour le vocabulaire et le vocabulaire final sera tous les ensembles découverts de bytes fréquents regroupés autant que possible tandis que les bytes qui ne sont pas fréquemment liés au même byte seront un token eux-mêmes.

## Data Sampling

Les LLMs comme GPT fonctionnent en prédisant le mot suivant en fonction des précédents, donc pour préparer des données pour l'entraînement, il est nécessaire de préparer les données de cette manière.

Par exemple, en utilisant le texte "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

Afin de préparer le modèle à apprendre à prédire le mot suivant (en supposant que chaque mot est un token utilisant le tokenizer très basique), et en utilisant une taille maximale de 4 et une fenêtre glissante de 1, voici comment le texte devrait être préparé :
```javascript
Input: [
["Lorem", "ipsum", "dolor", "sit"],
["ipsum", "dolor", "sit", "amet,"],
["dolor", "sit", "amet,", "consectetur"],
["sit", "amet,", "consectetur", "adipiscing"],
],
Target: [
["ipsum", "dolor", "sit", "amet,"],
["dolor", "sit", "amet,", "consectetur"],
["sit", "amet,", "consectetur", "adipiscing"],
["amet,", "consectetur", "adipiscing", "elit,"],
["consectetur", "adipiscing", "elit,", "sed"],
]
```
Notez que si la fenêtre glissante avait été de 2, cela signifie que la prochaine entrée dans le tableau d'entrée commencera 2 jetons après et non juste un, mais le tableau cible ne prédit toujours qu'un seul jeton. Dans pytorch, cette fenêtre glissante est exprimée dans le paramètre `stride`.
