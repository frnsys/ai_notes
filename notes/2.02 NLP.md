
# NLP

Much of this is adapted from my notes for the Natural Language Processing course taught by Dan Jurafsky and Christopher Manning at Stanford.

## Challenges

_Ambiguity_ is one of the greatest challenges to NLP:

For example:

> Fed raises interest rates, where "raises" is the verb, and "Fed" is the noun phrase
> Fed raises interest rates, where "interest" is the verb, and "Fed raises" is the noun phrase

Other challenges include:

- _non-standard english_: for instance, text shorthand, phrases such as "SOOO PROUD" as opposed to "so proud", or hashtags, etc
- _segmentation issues_: [the] [New] [York-New] [Haven] [Railroad] vs. [the] [New York]-[New Haven] [Railroad]
- _idioms_ (e.g. "get cold feet", doesn't literally mean what it says)
- _neologisms_ (e.g. "unfriend", "retweet", "bromance")
- _world knowledge_ (e.g. "Mary and Sue are sisters" vs "Mary and Sue are mothers.")
- _tricky entity names_: "Where is _A Bug's Life_ playing", or "a mutation on the _for_ gene"

The typical approach is to codify knowledge about language & knowledge about the world and find some way to combine them to build probabilistic models.

## Terminology

#### Synset
A synset is a set of synonyms that represent a single sense of a word.

#### Wordform
The full inflected surface form: e.g. "cat" and "cats" are different wordforms.

#### Lemma
The same stem, part of speech, rough word sense: e.g. "cat" and "cats" are the same lemma.

One lemma can have many meanings.

For example:

> a _bank_ can hold investments...
> agriculture on the east _bank_...

These usages have a different _sense_.

#### Sense

A discrete representation of an aspect of a word's meaning.

#### Homonyms

Words that share form but have unrelated, distinct meanings (such as "bank").

- _Homographs_: bank/bank, bat/bat
- _Homophones_: write/right, piece/peace

#### Polysemy

> the _bank_ was built in 1875 ("bank" = a building belonging to a financial institution)
> I withdrew money from the _bank_ ("bank" = a financial institution)

A _polysemous_ word has _related_ meanings.

_Systematic polysemy_, or _metonymy_, is when the meanings have a _systematic_ relationship.

For example, "school", "university", "hospital" - all can mean the institution or the building, so the systematic relationship here is `building <=> organization`.

Another example is `author <=> works of author`, e.g. "Jane Austen wrote Emma" and "I love Jane Austen".

#### Synonyms

Different words that have the same _propositional_ meaning in some or all contexts. However, there may be no examples of _perfect synonymy_ since even if propositional meaning is identical, they may vary in notions of politeness or other usages and so on.

For example, "water" and "H2O" - each are more appropriate in different contexts.

As another example, "big" and "large" - sometimes they can be swapped, sometimes they cannot:

> That's a big plane. How large is that plane? (Acceptable)
> Miss Nelson became kind of a big sister to Benjamin. Miss Nelson became kind of a large sister to Benjamin (Not as acceptable)

The latter works less because "big" has multiple senses, one of which does not correspond to "large".

#### Antonyms

Senses which are opposite with respect to one feature of meaning, but otherwise are similar, such as dark/light, short/fast, etc.

#### Hyponym

One sense is a hyponym of another if the first sense is more specific (i.e. denotes a subclass of the other).

- _car_ is a hyponym of _vehicle_
- _mango_ is a hyponym of _fruit_

#### Hypernym/Superordinate

- _vehical_ is a hypernym of _car_
- _fruit_ is a hypernym of _mango_

#### Token
An instance of that type in running text; $N$ = number of tokens, i.e. counting every word in the sentence, regardless of uniqueness.

#### Type
An element of the vocabulary; $V$ = vocabulary = set of types ($|V|$ = the size of the vocabulary), i.e. counting every unique word in the sentence.

## Data preparation

### Sentence segmentation

"!", "?" are pretty reliable indicators that we've reached the end of a sentence. Periods can mean the end of the sentence _or_ an abbreviation (e.g. Inc. or Dr.) or numbers (e.g. 4.3).

### Tokenization

The best approach for tokenization varies widely depending on the particular problem and language. German, for example, has many long compound words which you may want to split up. Chinese has no spaces (no easy way for _word segmentation_), Japanese has no spaces and multiple alphabets.

### Normalization

Once you have your tokens you need to determine how to normalize them. For example, "USA" and "U.S.A." could be collapsed into a single token. But about "Windows", "window", and "windows"?

Some common approaches include:

- _case folding_ - reducing all letters to lower case (but sometimes case may be informative)
- _lemmatization_ - reduce inflections or variant forms to base form.
- _stemming_ = reducing terms of their stems; a crude chopping of affixes; a simplified version of lemmatization. The Porter stemmer is the most common English stemmer.

### Term Frequency-Inverse Document Frequency (tf-idf) Weighting

Using straight word counts may not be the best approach in many cases.

Rare terms are typically more informative than frequent terms, so we want to bias our numerical representations of tokens to give rarer words higher weights. We do this via _inverse document frequency weighting_ (idf):

$$ idf_t = \log(\frac{N}{df_t}) $$

For a term $t$ which appears in $df$ documents ($df_t$ = document frequency for $t$).

$\log$ is used here to "dampen" the effect of idf.

This can be combined with $t$'s term frequency $tf_d$ for a particular document $d$ to produce tf-idf weighting, which is the best known weighting scheme for text information retrieval:

$$ w_{t,d} = (1 + \log tf_{t,d}) \times \log(\frac{n}{df_t}) $$

### The Vector Space Model (VSM)

This representation of text data - that is, some kind of numerical feature for each word, such as the tf-idf weight and frequency, defines a $|V|$-dimensional vector space (where $V$ is the vocabulary size).

- _terms_ are the axes of space
- _documents_ are points (vectors) in this space
- this space is _very high-dimensional_ when dealing with large vocabularies
- these vectors are very _sparse_ - most entries are zero

### Normalizing vectors

This is a different kind of normalization than the previously mentioned one, which was about normalizing the language. Here, we are normalizing vectors in a more mathematical sense.

Vectors can be length-normalized by dividing each of its components by its length. We can use the L2 norm, which makes it a _unit vector_ ("unit" means it is of length 1):

$$ ||\vec{x}||_2 = \sqrt{\sum_i x_i^2} $$

This means that if we have, for example, a document and copy of that document with every word doubled, length normalization causes each to have identical vectors (without normalization, the copy would have been twice as long).

## Measuring similarity between text

### Minimum edit distance

The _minimum edit distance_ between two strings is the minimum number of editing operations (insertion/deletion/substitution) needed to transform one into the other. Each editing operation has a cost of 1, although in _Levenshtein minimum edit distance_ substitutions cost 2 because they are composed of a deletion and an insertion.

### Jaccard coefficient

The Jaccard coefficient is a commonly-used measure of overlap for two sets $A$ and $B$.

$$ jaccard(A,B) = \frac{|A \cap B|}{|A \cup B|} $$

A set has a Jaccard coefficient of 1 against itself: $jaccard(A,A) = 1$.

If $A$ and $B$ have no overlapping elements, $jaccard(A,B) = 0$.

The Jaccard coefficient does _not_ consider term frequency, just set membership.

### Euclidean Distance

Using the vector space model above, the similarity between two documents can be measured by the euclidean distance between their two vectors.

However, euclidean distance can be problematic since longer vectors have greater distance.

For instance, there could be one document vector, $a$, and another document vector $b$ which is just a scalar multiple of the first document. Intuitively they may be more similar since they lie along the same line. But by euclidean distance, $c$ is closer to $a$.

![Euclidean distances](assets/euclidean_distance.svg)

### Cosine similarity

In cases like the euclidean distance example above, using _angles_ between vectors can be a better metric for similarity.

For length-normalized vectors, cosine similarity is just their dot product:

$$ cos(\vec{q}, \vec{d}) = \vec{q} \cdot \vec{d} = \sum_{i=1}^{|V|} q_i d_i $$

Where $q$ and $d$ are length-normalized vectors and $q_i$ is the tf-idf weight of term $i$ in document $q$ and $d_i$ is the tf-idf weight of term $i$ in document $d$.

## Probabilistic Language Models

The approach of probabilistic language models involves generating some probabilistic understanding of language - what is likely or unlikely.

These probabilistic models have applications in many areas:

- Machine translation:
    $P(\text{high winds tonight}) > P(\text{large winds tonight})$.
- Spelling correction:
    $P(\text{about fifteen minutes from}) > P(\text{about fifteen minuets from})$.
- Speech recognition:
    $P(\text{I saw a van}) > P(\text{eyes awe of an})$.

So generally you are asking: what is the probability of this given sequence of words?

You could use the _chain rule_ here:

$$
\begin{aligned}
P(\text{the water is so transparent}) = \\
P(\text{the}) \times P(\text{water}|\text{the}) \times P(\text{is}|\text{the water}) \\
\times P(\text{so}|\text{the water is}) \times P(\text{transparent}|\text{the water is so})
\end{aligned}
$$

Formally, the above would be expressed:

$$ P(w_1 w_2 \dots w_n) = \prod_i P(w_i|w_1 w_2 \dots w_{i-1}) $$

Note that probabilities are usually done in _log space_ to avoid _underflow_, which occurs if you're multiplying many small probabilities together, and because then you can just add the probabilities, which is faster than multiplying:

$$ p_1 \times p_2 \times p_3 = \log p_1 + \log p_2 + \log p_3 $$

To make estimating these probabilities manageable, we use the _Markov assumption_ and assume that a given word's conditional probability only depends on the immediately preceding $k$ words, _not_ the entire preceding sequence:

$$ P(w_1 w_2 \dots w_n) \approx \prod_i P(w_i| w_{i-k} \dots w_{i-1}) $$

### n-grams

The _unigram_ model treats each word as if they have an independent probability:

$$ P(w_1 w_2 \dots w_n) \approx \prod_i P(w_i) $$

The _bigram_ model conditions on the previous word:

$$ P(w_1 w_2 \dots w_{i-1}) \approx \prod_i P(w_i | w_{i-1}) $$

We estimate bigram probabilities using the _maximum likelihood estimate_ (MLE):

$$ P_{MLE}(w_i | w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})} $$

Which is just the count of word $i$ occuring after word $i-1$ over all of the occurences of word $i-1$.

This can be extended to trigrams, 4-grams, 5-grams, etc.

Though language has _long-distance dependencies_, i.e. the probability of a word can depend on another word much earlier in the sentence, n-grams work well in practice.

#### Dealing with zeros

Zeroes occur if some n-gram occurs in the testing data which didn't occur in the training set.

Say we had the following training set:

> ... denied the reports
> ... denied the claims
> ... denied the request

And the following test set:

> ... denied the offer

Here $P(\text{offer} | \text{denied the}) = 0$ since the model has not encountered that term.

We can get around this using _Laplace smoothing_, also known as _add-one smoothing): simply pretend that we saw each word once more than we actually did (i.e. add one to all counts).

With add-one smoothing, our MLE becomes:

$$ P_{Add-1}(w_i | w_{i-1}) = \frac{count(w_{i-1}, w_i) + 1}{count(w_{i-1}) + V} $$

Note that this smoothing can be very blunt and may drastically change your counts.

#### Interpolation

With interpolation, you mix unigrams, bigrams, and trigrams, assigning weights to each.

Simple linear interpolation looks like:

$$ P(w_n | w_{n-1} w_{n-2}) = \lambda_1 P (w_n | w_{n-1} w_{n-2}) + \lambda_2 P(w_n|w_{n-1}) + \lambda_3 P(w_n) $$

Such that $\sum_i \lambda_i = 1$. The $\lambda$ parameters here are the different weights for the different n-grams.

If you want to get more complex, you can have the $\lambda$ parameters vary by context:

$$ P(w_n | w_{n-1} w_{n-2}) = \lambda_1(w^{n-1}_{n-2}) P (w_n | w_{n-1} w_{n-2}) + \lambda_2(w^{n-1}_{n-2}) P(w_n|w_{n-1}) + \lambda_3(w^{n-1}_{n-2}) P(w_n) $$

The process of determining the $\lambda$ parameters usually involves holding out part of the training corpus, then getting the n-gram probabilities from the remaining training data, then selecting the $\lambda$ parameters to maximize the probability of the held-out data:

$$ \log P(w_1 \dots w_n | M(\lambda_1 \dots \lambda_k)) = \sum \log P_{M(\lambda_1 \dots \lambda_k)} (w_i | w_{i-1}) $$

## Text Classification

The general text classification problem is given an input document $d$ and a fixed set of classes $C = \{c_1, c_2, \dots, c_j\}$ output a predicted class $c \in C$.

### Naive Bayes

This supervised approach to classification is based on Bayes' rule. It relies on a very simple representation of the document called "bag of words", which is ignorant of the sequence or order of word occurrence (and other things), and only pays attention to their counts/frequency.

So you can represent the problem with Bayes' rule:

$$
P(c|d) = \frac{P(d|c)P(c)}{P(d)}
$$

And the particular problem at hand is finding the class which maximizes $P(c|d)$, that is:

$$
\begin{aligned}
C_{MAP} &= \text{argmax}_{c \in C} P(c|d)
&= \text{argmax}_{c \in C} P(d|c)P(c)
\end{aligned}
$$

Where $C_{MAP}$ is the maximum a posteriori class.

Using our bag of words assumption, we represent a document as features $x_1, \dots x_n$ without concern for their order:

$$ C_{MAP} = \text{argmax}_{c \in C} P(x_1, x_2, \dots, x_n | c)P(c) $$

We additionally assume _conditional independence_, i.e. that the presence of one word doesn't have any impact on the probability of any other word's occurrence:

$$ P(x_1, x_2, \dots, x_n | c) = P(x_1 | c) \cdot P(x_2 | c) \cdot \dots \cdot P(x_n | c) $$

And thus we have the _multinomial naive bayes classifier_:

$$ C_{NB} = \text{argmax}_{c \in C} P(c_j) \prod_{x \in X} P(x | c) $$

To calculate the prior probabilities, we use the _maximum likelihood estimates_ approach:

$$ P(c_j) =  \frac{\text{doccount}(C = c_j)}{N_{doc}} $$

That is, the prior probability for a given class is the count of documents in that class over the total number of documents.

Then, for words:

$$ P(w_i | c_j) = \frac{count(w_i, c_j)}{\sum_{w \in V}count(w, c_j)} $$

That is, the count of a word in documents of a given class, over the total count of words in that class.

To get around the problem of zero probabilities (for words encountered in test input but not in training, which would cause a probability of a class to be zero since the probability of a class is the joint probability of the words encountered), you can use Laplace smoothing (see above):

$$ P(w_i | c_j) = \frac{count(w_i, c_j) + 1}{(\sum_{w \in V}count(w, c_j)) + |V|} $$

Note that to avoid underflow (from multiplying lots of small probabilities), you may want to work with log probabilities (see above).

In practice, even with all these assumptions, Naive Bayes can be quite good:

- Very fast, low storage requirements
- Robust to irrelevant features (they tend to cancel each other out)
- Very good in domains with many equally important features
- Optimal if independence assumptions hold
- A good, dependable baseline for text classification

### Evaluating text classification

The possible outcomes are:

- true positive: correctly identifying something as true
- false positive: incorrectly identifying something as true
- true negative: correctly identifying something as false
- false negative: incorrectly identifying something as false

The _accuracy_ of classification is calculated as:

$$ \text{accuracy} = \frac{tp + tn}{tp + fp + fn + fn} $$

Though as a metric it isn't very useful if you are dealing with situations where the correct class is sparse and most words you encounter are not in the correct class:

> Say you're looking for a word that only occurs 0.01% of the time. you have a classifier you run on 100,000 docs and the word appears in 10 docs (so 10 docs are correct, 99,990 are not correct). but you can have that classifier classify all docs as not correct and get an amazing accuracy of 99,990/100,000 = 99.99% but the classifier didn't actually do anything!

So other metrics are needed.

_Precision_ measures the percent of selected items that are correct:

$$ \text{precision} = \frac{tp}{tp + fp} $$

_Recall_ measures the percent of correct items that are selected:

$$ \text{recall} = \frac{tp}{tp = fn} $$

Typically, there is a trade off between recall and precision - the improvement of one comes at the sacrifice of the other.

The _F measure_ combines both precision and recall into a single metric:

$$ F = \frac{1}{\alpha \frac{1}{P} + (1-\alpha) \frac{1}{R}} = \frac{(\beta^2 + 1)PR}{\beta^2 P + R} $$

Where $\alpha$ is a weighting value so you can assign more importance to either precision or recall.

People usually use the _balanced F1 measure_, where $\beta = 1$ (that is, $\alpha = 1/2$):

$$ F = \frac{2PR}{P+R} $$




## Named Entity Recognition (NER)

Named entity recognition is the extraction of _entities_ - people, places, organizations, etc - from a text.

Many systems use a combination of statistical techniques, linguistic parsing, and gazetteers to maximize detection recall & precision.  Distant supervision and unsupervised techniques can also help with training, limiting the amount of gold-standard data necessary to build a statistical model.

_Boundary errors_ are common in NER:

> First _Bank of Chicago_ announced earnings...

Here, the extractor extracted "Bank of Chicago" when the correct entity is the "First Bank of Chicago".

A general NER approach is to use supervised learning:

1. Collect a set of training documents
2. Label each entity with its entity type or `O` for "other".
3. Design feature extractors
4. Train a sequence classifier to predict the labels from the data.

## Relation Extraction

> International Business Machines Corporation (IBM or the company) was incorporate in the State of New York on June 16, 1911, as the Computing-Tabulating-Recording Co. (C-T-R)...

From such a text you could extract the following _relation triples_:

    Founder-year(IBM,1911)
    Founding-location(IBM,New York)

These relations may be represented as _resource description framework (RDF) triples_ in the form of `subject predicate object`.

> Golden Gate Park location San Francisco

### Ontological Relations

- `IS-A` describes a subsumption between classes, called a _hypernum_:

    > Giraffe IS-A ruminant IS-A ungulate IS-A mammal IS-A vertebrate IS-A animal...

- `instance-of` relation between individual and class

    > San Francisco instance-of city

There may be many domain-specific ontological relations as well, such as `founded` (between a `PERSON` and an `ORGANIZATION`), `cures` (between a `DRUG` and a `DISEASE`), etc.

### Methods

Relation extractors can be built using:

- handwritten patterns
- supervised machine learning
- semi-supervised and unsupervised
    - bootstrapping (using seeds)
    - distance supervision
    - unsupervised learning from the web

#### Handwritten patterns

- Advantages:
    - can take advantage of domain expertise
    - human patterns tend to be high-precision
- Disadvantages:
    - human patterns are often low-recall
    - hard to capture all possible patterns

#### Supervised

- Advantages:
    - can get high accuracy if...
        - there's enough hand-labeled training data
        - if the test is similar enough to training
- Disadvantages:
    - labeling a large training set is expensive
    - don't generalize well

You could use classifiers: find all pairs of named entities, then use a classifier to determine if the two are related or not.

#### Unsupervised

If you have no training set and either only a few seed tuples or a few high-precision patterns, you can _bootstrap_ and use the seeds to accumulate more data.

The general approach is:

1. Gather a set of seed pairs that have a relation $R$
2. Iterate:
    1. Find sentences with these pairs
    2. Look at the context between or around the pair
    3. Generalize the context to create patterns
    4. Use these patterns to find more pairs

For example, say we have the seed tuple $<\text{Mark Twain, Elmira}>$. We could use Google or some other set of documents to search based on this tuple. We might find:

- "Mark Twain is buried in Elmira, NY"
- "The grave of Mark Twain is in Elmira"
- "Elmira is Mark Twain's final resting place"

which gives us the patterns:

- "X is buried in Y"
- "The grave of X is in Y"
- "Y is X's final resting place"

Then we can use these patterns to search and find more tuples, then use those tuples to find more patterns, etc.

Two algorithms for this bootstrapping is the Dipre algorithm and the Snowball algorithm, which is a version of Dipre which requires the strings be named entities rather than any string.

Another semi-supervised algorithm is _distance supervision_, which mixes bootstrapping and supervised learning. Instead of a few seeds, you use a large database to extract a large number of seed examples and go from there:

1. For each relation $R$
2. For each tuple in a big database
3. Find sentences in a large corpus with both entities of the tuple
4. Extract frequent contextual features/patterns
5. Train a supervised classifier using the extracted patterns

## Sentiment Analysis

In general, sentiment analysis involves trying to figure out if a sentence/doc/etc is positive/favorable or negative/unfavorable; i.e. detecting _attitudes_ in a text.

The attitude may be

- a simple weighted polarity (positive, negative, neutral), which is more common
- from a set of types (like, love, hate, value, desire, etc)

When using multinomial Naive Bayes for sentiment analysis, it's often better to use _binarized_ multinomial Naive Bayes under the assumption that word occurrence matters more than word frequency: seeing "fantastic" five times may not tell us much more than seeing it once. So in this version, you would cap word frequencies at one.

An alternate approach is to use $\log(freq(w))$ instead of 1 for the count.

However, sometimes raw word counts don't work well either. In the case of IMDB ratings, the word "bad" appears in more 10-star reviews than it does in 2-star reviews!

Instead, you'd calculate the _likelihood_ of that word occurring in an $n$-star review:

$$
P(w|c) = \frac{f(w,c)}{\sum_{w \in C} f(w,c)}
$$

And then you'd used the _scaled likelihood_ to make these likelihoods comparable between words:

$$
\frac{P(w|c)}{P(w)}
$$

### Sentiment Lexicons

Certain words have specific sentiment; there are a variety of sentiment lexicons which specify those relationships.

### Challenges

#### Negation

"I _didn't_ like this movie" vs "I _really_ like this movie."

One way to handle negation is to prefix every word following a negation word with `NOT_`, e.g. "I didn't NOT_like NOT_this NOT_movie".

#### "Thwarted Expectations" problem

For example, a film review which talks about how great a film _should_ be, but fails to live up to those expectations:

> This film should be _brilliant_. It sounds like a _great_ plot, the actors are _first grade_, and the supporting cast is _good_ as well, and Stallone is attempting to deliver a good performance. However, it _can't hold up_.

## Summarization

Generally, sumamrization is about producing an abridged version of a text without or with minimal loss of important information.

There are a few ways to categorize summarization problems.

- Single-document vs multi-document summarization: summarizing a single document, yielding an abstract or outline or headline, or producing a gist of the content of multiple documents?
- Generic vs query-focused summarization: give a general summary of the document, or a summary tailored to a particular user query?
- Extractive vs abstractive: create a summary from sentences pulled from the document, or generate new text for the summary?

Here, extractive summarization will be the focus (abstractive summarization is really hard).

The _baseline_ used in summarization, which often works surprisingly well, is just to take the first sentence of a document.

### The general approach

Summarization usually uses this process:

1. Content Selection: choose what sentences to use from the document.
    - You may weight salient words based on tf-idf, its presence in the query (if there is one), or based on topic signature.
        - For the latter, you can use _log-likelihood ratio_ (LLR):

            $$ weight(w_i) = \begin{cases} 1 & \text{if} -2\log\lambda(w_i)>10 \\ 0 & \text{otherwise} \end{cases} $$

    - Weight a sentence (or a part of a sentence, i.e. a _window_) by the weights of its words:

        $$ weight(s) = \frac{1}{|S|} \sum_{w \in S} weight(w) $$

    - You can combine LLR with _maximal marginal relevance_ (MMR), which is a greedy algorithm which selects sentences by their similarity to the query and by their dissimilarity (novelty) to already-selected sentences to avoid redundancy.

2. Information Ordering: choose the order for the sentences in the summary.
    - If you are summarizing documents with some chronological order to them, such as the news, then it makes sense to order sentences chronologically (if you are, for example, summarizing a set of news articles).
    - You can also use _topical ordering_, and order sentences by the order of topics in the source documents.
    - You can also use _coherence_:
        - Choose orderings that make neighboring sentences (cosine) similar.
        - Choose orderings in which neighboring sentences discuss the same entity.

3. Sentence Realization: clean up the sentences so that the summary is coherent or remove unnecessary content. You could remove:
    - _appositives_: "Rajam[, an artist living in Philadelphia], found inspiration in the back of city magazines."
    - _attribution clauses_: "Sources said Wednesday"
    - _initial adverbials_: "For example", "At this point"
