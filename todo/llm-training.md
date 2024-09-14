# LLM Training

## Tokenizing

Tokenizing consists on separating the data in specific chunks and assign them specific IDs (numbers).\
A very simple tokenizer for texts might to just get each word of a text separately, and also punctuation symbols and remove spaces.\
Therefore, `"Hello, world!"` would be: `["Hello", ",", "world", "!"]`

Then, in order to assign each of the words and symbols a token ID (number), it's needed to create the tokenizer **vocabulary**. If you are tokenizing for example a book, this could be **όλες οι διαφορετικές λέξεις του βιβλίου** σε αλφαβητική σειρά με μερικούς επιπλέον τόκεν όπως:

* `[BOS] (Αρχή της ακολουθίας)`: Τοποθετείται στην αρχή ενός κειμένου, υποδεικνύει την αρχή ενός κειμένου (χρησιμοποιείται για να διαχωρίσει μη σχετιζόμενα κείμενα).
* `[EOS] (Τέλος της ακολουθίας)`: Τοποθετείται στο τέλος ενός κειμένου, υποδεικνύει το τέλος ενός κειμένου (χρησιμοποιείται για να διαχωρίσει μη σχετιζόμενα κείμενα).
* `[PAD] (padding)`: Όταν το μέγεθος της παρτίδας είναι μεγαλύτερο από ένα (συνήθως), αυτό το τόκεν χρησιμοποιείται για να αυξήσει το μήκος αυτής της παρτίδας ώστε να είναι όσο μεγαλύτερη είναι οι άλλες.
* `[UNK] (άγνωστο)`: Για να αναπαραστήσει άγνωστες λέξεις.

Following the example, having tokenized a text assigning each word and symbol of the text a position in the vocabulary, the tokenized sentence `"Hello, world!"` -> `["Hello", ",", "world", "!"]` would be something like: `[64, 455, 78, 467]` supposing that `Hello` is at pos 64, "`,"` is at pos `455`... in the resulting vocabulary array.

However, if in the text used to generate the vocabulary the word `"Bye"` didn't exist, this will result in: `"Bye, world!"` -> `["[UNK]", ",", "world", "!"]` -> `[987, 455, 78, 467]` supposing the token for `[UNK]` is at 987.

### BPE - Byte Pair Encoding

In order to avoid problems like needing to tokenize all the possible words for texts, LLMs like GPT used BPE which basically **κωδικοποιεί συχνά ζεύγη byte** για να μειώσει το μέγεθος του κειμένου σε μια πιο βελτιστοποιημένη μορφή μέχρι να μην μπορεί να μειωθεί περισσότερο (check [**wikipedia**](https://en.wikipedia.org/wiki/Byte\_pair\_encoding)). Note that this way there aren't "unknown" words for the vocabulary and the final vocabulary will be all the discovered sets of frequent bytes together grouped as much as possible while bytes that aren't frequently linked with the same byte will be a token themselves.

## Data Sampling

LLMs like GPT work by predicting the next word based on the previous ones, therefore in order to prepare some data for training it's neccesary to prepare the data this way.

For example, using the text "Lorem ipsum dolor sit amet, consectetur adipiscing elit,"

In order to prepare the model to learn predicting the following word (supposing each word is a token using the very basic tokenizer), and using a max size of 4 and a sliding window of 1, this is how the text should be prepared:
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
Σημειώστε ότι αν το sliding window ήταν 2, αυτό σημαίνει ότι η επόμενη είσοδος στον πίνακα εισόδου θα ξεκινούσε 2 tokens μετά και όχι μόνο ένα, αλλά ο στόχος πίνακας θα συνεχίσει να προβλέπει μόνο 1 token. Στο pytorch, αυτό το sliding window εκφράζεται στην παράμετρο `stride`.
