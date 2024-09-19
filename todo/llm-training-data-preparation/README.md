# LLM Training - Data Preparation

**Αυτές είναι οι σημειώσεις μου από το πολύ προτεινόμενο βιβλίο** [**https://www.manning.com/books/build-a-large-language-model-from-scratch**](https://www.manning.com/books/build-a-large-language-model-from-scratch) **με κάποιες επιπλέον πληροφορίες.**

## Basic Information

Πρέπει να ξεκινήσετε διαβάζοντας αυτή την ανάρτηση για κάποιες βασικές έννοιες που πρέπει να γνωρίζετε:

{% content-ref url="0.-basic-llm-concepts.md" %}
[0.-basic-llm-concepts.md](0.-basic-llm-concepts.md)
{% endcontent-ref %}

## 1. Tokenization

{% hint style="success" %}
Ο στόχος αυτής της αρχικής φάσης είναι πολύ απλός: **Διαιρέστε την είσοδο σε tokens (ids) με κάποιον τρόπο που έχει νόημα**.
{% endhint %}

{% content-ref url="1.-tokenizing.md" %}
[1.-tokenizing.md](1.-tokenizing.md)
{% endcontent-ref %}

## 2. Data Sampling

{% hint style="success" %}
Ο στόχος αυτής της δεύτερης φάσης είναι πολύ απλός: **Δειγματοληψία των δεδομένων εισόδου και προετοιμασία τους για τη φάση εκπαίδευσης, συνήθως διαχωρίζοντας το σύνολο δεδομένων σε προτάσεις συγκεκριμένου μήκους και δημιουργώντας επίσης την αναμενόμενη απάντηση.**
{% endhint %}

{% content-ref url="2.-data-sampling.md" %}
[2.-data-sampling.md](2.-data-sampling.md)
{% endcontent-ref %}

## 3. Token Embeddings

{% hint style="success" %}
Ο στόχος αυτής της τρίτης φάσης είναι πολύ απλός: **Αναθέστε σε κάθε από τα προηγούμενα tokens στο λεξιλόγιο έναν διανύσμα των επιθυμητών διαστάσεων για να εκπαιδεύσετε το μοντέλο.** Κάθε λέξη στο λεξιλόγιο θα είναι ένα σημείο σε έναν χώρο X διαστάσεων.\
Σημειώστε ότι αρχικά η θέση κάθε λέξης στο χώρο είναι απλώς αρχικοποιημένη "τυχαία" και αυτές οι θέσεις είναι εκπαιδεύσιμες παράμετροι (θα βελτιωθούν κατά τη διάρκεια της εκπαίδευσης).

Επιπλέον, κατά τη διάρκεια της ενσωμάτωσης tokens **δημιουργείται ένα άλλο επίπεδο ενσωματώσεων** που αντιπροσωπεύει (σε αυτή την περίπτωση) τη **απόλυτη θέση της λέξης στην προτασούλα εκπαίδευσης**. Με αυτόν τον τρόπο, μια λέξη σε διαφορετικές θέσεις στην πρόταση θα έχει διαφορετική αναπαράσταση (νόημα).
{% endhint %}

{% content-ref url="3.-token-embeddings.md" %}
[3.-token-embeddings.md](3.-token-embeddings.md)
{% endcontent-ref %}

## 4. Attention Mechanisms

{% hint style="success" %}
Ο στόχος αυτής της τέταρτης φάσης είναι πολύ απλός: **Εφαρμόστε κάποιους μηχανισμούς προσοχής**. Αυτοί θα είναι πολλά **επανερχόμενα επίπεδα** που θα **συλλαμβάνουν τη σχέση μιας λέξης στο λεξιλόγιο με τους γείτονές της στην τρέχουσα πρόταση που χρησιμοποιείται για την εκπαίδευση του LLM**.\
Χρησιμοποιούνται πολλά επίπεδα γι' αυτό, οπότε πολλές εκπαιδεύσιμες παράμετροι θα συλλαμβάνουν αυτές τις πληροφορίες.
{% endhint %}

{% content-ref url="4.-attention-mechanisms.md" %}
[4.-attention-mechanisms.md](4.-attention-mechanisms.md)
{% endcontent-ref %}

## 5. LLM Architecture

{% hint style="success" %}
Ο στόχος αυτής της πέμπτης φάσης είναι πολύ απλός: **Αναπτύξτε την αρχιτεκτονική του πλήρους LLM**. Συνδυάστε τα πάντα, εφαρμόστε όλα τα επίπεδα και δημιουργήστε όλες τις λειτουργίες για να παράγετε κείμενο ή να μετατρέπετε κείμενο σε IDs και αντίστροφα.

Αυτή η αρχιτεκτονική θα χρησιμοποιηθεί και για την εκπαίδευση και για την πρόβλεψη κειμένου μετά την εκπαίδευση.
{% endhint %}

{% content-ref url="5.-llm-architecture.md" %}
[5.-llm-architecture.md](5.-llm-architecture.md)
{% endcontent-ref %}

## 6. Pre-training & Loading models

{% hint style="success" %}
Ο στόχος αυτής της έκτης φάσης είναι πολύ απλός: **Εκπαιδεύστε το μοντέλο από την αρχή**. Για αυτό θα χρησιμοποιηθεί η προηγούμενη αρχιτεκτονική LLM με κάποιους βρόχους που θα διατρέχουν τα σύνολα δεδομένων χρησιμοποιώντας τις καθορισμένες συναρτήσεις απώλειας και τον βελτιστοποιητή για να εκπαιδεύσουν όλες τις παραμέτρους του μοντέλου.
{% endhint %}

{% content-ref url="6.-pre-training-and-loading-models.md" %}
[6.-pre-training-and-loading-models.md](6.-pre-training-and-loading-models.md)
{% endcontent-ref %}

## 7.0. LoRA Improvements in fine-tuning

{% hint style="success" %}
Η χρήση του **LoRA μειώνει πολύ την υπολογιστική ισχύ** που απαιτείται για να **βελτιώσετε** ήδη εκπαιδευμένα μοντέλα.
{% endhint %}

{% content-ref url="7.0.-lora-improvements-in-fine-tuning.md" %}
[7.0.-lora-improvements-in-fine-tuning.md](7.0.-lora-improvements-in-fine-tuning.md)
{% endcontent-ref %}

## 7.1. Fine-Tuning for Classification

{% hint style="success" %}
Ο στόχος αυτής της ενότητας είναι να δείξει πώς να βελτιώσετε ένα ήδη προεκπαιδευμένο μοντέλο έτσι ώστε αντί να παράγει νέο κείμενο, το LLM θα επιλέγει να δώσει τις **πιθανότητες του δεδομένου κειμένου να κατηγοριοποιηθεί σε κάθε μία από τις δεδομένες κατηγορίες** (όπως αν ένα κείμενο είναι spam ή όχι).
{% endhint %}

{% content-ref url="7.1.-fine-tuning-for-classification.md" %}
[7.1.-fine-tuning-for-classification.md](7.1.-fine-tuning-for-classification.md)
{% endcontent-ref %}

## 7.2. Fine-Tuning to follow instructions

{% hint style="success" %}
Ο στόχος αυτής της ενότητας είναι να δείξει πώς να **βελτιώσετε ένα ήδη προεκπαιδευμένο μοντέλο για να ακολουθεί οδηγίες** αντί να παράγει απλώς κείμενο, για παράδειγμα, απαντώντας σε καθήκοντα ως chatbot.
{% endhint %}

{% content-ref url="7.2.-fine-tuning-to-follow-instructions.md" %}
[7.2.-fine-tuning-to-follow-instructions.md](7.2.-fine-tuning-to-follow-instructions.md)
{% endcontent-ref %}
