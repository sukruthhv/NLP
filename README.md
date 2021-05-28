![enter image description here](https://appen.com/wp-content/uploads/2020/10/Natural-Language-Processing-NLP.png)
<h2>  NLP (Natural Language Processing) </h2>
This is a NLP repo, where-in in i like to share my learnings about NLP and the various concepts involved in NLP. NLP has quite a lot of applications, which are quite useful and intuitive. For ex:- Think about Chatbots, Recommendation systems, sentiment analysis and many more, where the user input (predominantly text) has been taken and processed to feed the ML models and Neural networks to have the desried prediction and other outcomes.

In this repo, i am going to share some of my insights about the following concepts :

 1. Need of Language processing
 2. Some importnat Nomenclature of NLP
 3. Bag of Words / Count Vectorizer
 4. TF-IDF (Term Frequency - Inverse Document Frequency)
 5. Word2Vec

<h3> Need of Language processing </h3>
First of all, we may need to understand as what are the potential applications, where teh NLP would be used, as this serves as a motivation for us to move forward and develop our interest towards NLP.

These are some of the amazing usecases, where NLP has been used :
1. **Chatbots** : Every application or support channel now has implemented the Chatbots, which help reduce the overall human support infrastructure, where-in most of the comonly asked FAQ's would be answered by the chatbots, which in turn reduce the overall burden for the support assistants in teh back-end. This is NLP used to process and analyze the queries asked by teh customers and provide relevant answers for these queries.
2. **Hiring** : Its a humongous task for HR teams to find teh right resume with right skills for a group in a company by screening 100's and 1000's of resume's. Where NLP can be used to screen all the resume's and provide teh relevant one's based on the desired requirements from recruiters.
3. **Voice assistants** : Majority of the people have been using Voice assistants liek Siri, Cortona, Ok Google and Alexa one or teh other way. NLP also plays a vital role here to analyze and help gather the right information to the users.
4. **Spelling and Grammer checker applications** : One of the important features of any word / text applications, which help us auto-correct our grammatical mistakes, which is again can be achieved with NLP easily.

These are just some of the sample applications to name, however there are numerous applications that we use in our daily lives that has been using NLP underneath.

<h3> Nomenclature used in NLP</h3>

Now, lets jump in and try to understand some common nomenclature in NLP :

Before we start with the Models, lets try to understand as what an ***Word embedding*** mean?

***"Word embedding"*** is the concept of converting words to vectors/numbers. This is quite important as the modles can work with Numerical data.

There are 2 types of embeddings :

 ***1. Frequency based embedding
 2. Prediction based embedding***

 - Corpus : The set of text / words used in a sentence or an article is called corpus. For ex :- If you have a PDF document with 1000 words, then these 1000 words forms a corpus.
 - Vocabulary : The unique set of words from a corpus is called a vocabulary.
 - Tokenization : is a process of dividing the sentence or text into tokens, where-in these tokens can be characters, words or sub-words. For ex :- if the sentence is "Tokenization is the smarter technique in NLP", then this sentence can be tokenized for characters like 'T','o','k','e','n'......., where as if is tokenized for words, then we get "Tokenization,"is","the","smarter"...., However if we tokenize for sub-words, we get something like smarter can become smart and er.
 - Stop words : These are the commonly used binding words like is, the, and, this, or, that and so on in a given language grammer.

<h3> Bag of Words / Count Vectorizer </h3>

Now that we have some understanding about the nomenclature above, let's get to our first concept in NLP called Bag of Words. In this model, we try to find the corpus from teh document and then find teh frequency of each of the item in corpus. For instance, Let's say we have 2 documents with teh following text :
Sentence1 : "This is the github blog about NLP"
Sentence2 : "This blog is very interesting, infact NLP itself is very interesting"

Now the Bag of words model would basically create the frequency of each of teh words in these documents as follows :
|Words|Frequency  |
|--|--|
| This |2|
| is |3  |
|the|1|
|github  |1  |
|blog|2|
|NLP|2  |

As we can see, its basically create a frequency vector for each of the words above in the corpus. However we could see that the general english vocabulary/binding words seems to be having more frequency or weightage among the 2 documents/sentences right? Hence we might need to reduce the significance of these binding words to analyze and give more importance to important words that matter most like NLP, github and so on. We can remove stop words and hence provide more weightage to othert words. 

How do we do it? That is where the ***NLTK library*** would be useful, which helps in the removal of stop words of a given language. "***CountVectorizer***" module/function is part of the sklearn which helps in achieving the frequency of all words in the corpus.

<h4> Stemming and Lemmatization</h4>
Before procedding to TF-IDF , lets discuss about Stemming and Lemmatization. 

Source :- http://alturl.com/92ev4 

The concept of stemming and lemmatization is very well explained in detail in the above source. However i am just providing the jist of it over here.

Languages we speak and write are made up of several words often derived from one another. When a language contains words that are derived from another word as their use in the speech changes is called **Inflected Language**. As you have read the definition of inflection with respect to grammar, you can understand that an inflected word(s) will have a common root form. Let's look at a few examples :

Playing, Plays, Played have a common root word called "Play", however the words Playing, Plays and Played are called ***Inflected words.*** However the infected words would provide the same/similar meaning as the root words like play here instead Playing and Played.

>  "Stemming is the process of reducing inflection in words to their
> root forms such as mapping a group of words to the same stem even if
> the stem itself is not a valid word in the Language."

Stemming is great and faster, however the one disadvantage of Stemming is that, it doesn't really care about the actual english word while stemming. For ex : when we stem "troubling", "troubled", "troubles", then we see a stemmed word called "troubl"

    from nltk.stem import PorterStemmer
    pr=PorterStemmer()
    
    pr.stem('troubling')
    
    pr.stem('troubled')
    
    pr.stem('troubles')
    
    o/p: 'troubl'
  
  Now, this doesn't make any sense as per the actual english vocabulary. Hence stemming cannot be used in all the usecases, like in chatbots, where the proper vocabulary is important.
  
That is where ***"Lemmatization"*** comes to our rescue. The concept remains the same, however 

> Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language.

    from nltk.stem import WordNetLemmatizer
    lemma=WordNetLemmatizer()
    
    lemma.lemmatize('troubling',pos='v')
    o/p: trouble
    
    lemma.lemmatize('troubles',pos='v')
    o/p: trouble
    
    lemma.lemmatize('troubled',pos='v')
    o/p:trouble

> "Observe the "pos" parameter passed to the lemmatize function, which
> is basically telling the lemmatizer that the Parts-of-speech = verb.
> You can also provide Noun and so on."

<h3> TF-IDF (Term-Frequency and Inverse Document Frequency)</h3>
Now that we have understood the concept of Bag of words, we see that the BOW model would basically take the frequency of each word in corpus. Lets take an example about the restaurant review system :

> Example: If we are given 4 reviews for an Italian pasta dish.
> 
> Review 1 : This pasta is very tasty and affordable.
> 
> Review 2: This pasta is not tasty and is affordable.
> 
> Review 3 : This pasta is delicious and cheap.
> 
> Review 4: Pasta is tasty and pasta tastes good.

Now if we find the weightage of word/term "tasty" using BoW, then we get the following :

**Review 1:**
| Words | Frequency |
|--|--|
|  Pasta|1  |
|  Tasty|1  |
|  Affordable|1  |

**Review 2:**

| Words | Frequency |
|--|--|
|  Pasta|1  |
|  Tasty|1  |
|  Affordable|1  |
|  Not|1  |

**Review 4:**
| Words | Frequency |
|--|--|
|  Pasta|1  |
|  Tasty|1  |
|  good|1  |

As per the BoW model, we see that both review 1 2 and 4 have the same weightage based on the word "Tasty"? But actually speaking the Review 2 is contrast towards the Review 1 and 4, which says that the food is not tasty. Hence BoW would fail in these scenario's. That is when the TF-IDF would come for a rescue.

> **tf–idf** or **TFIDF**, short for **term frequency-inverse document frequency**, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general.

The formula is simple :

> TF = No. of times a word/term occur in that document / Total No. of documents 
> IDF = log(Total No. of documents / No. of documents in which the Term of interest occur)

TD-IDF = TF * IDF

Now, if we take the same above example, then if we calc the TF-IDF for term "tasty" in each of teh document (Document here is each review). 

**Review 1:**

TF = (1/7) = 0.1428 ;  IDF = log(4/3) = 0.1249   [ Note : its log to base 10 - Natural log]
TF-IDF=0.1428*0.1249 = 0.0178

**Review 2:**

TF = (1/8) = 0.125 ;  IDF = log(4/3) = 0.1249
TF-IDF=0.125*0.1249 = 0.0156

**Review 4:**

TF = (1/7) = 0.125 ;  IDF = log(4/3) = 0.1249
TF-IDF=0.125*0.1249 = 0.0178

Here we coudl clearly see that the importance of word tasty is more in Review 1 and Review 4 compared to that of 2!!!! These are the cases, where the TF-IDF outshines the BoW model.

However 1 disadvantage we still live with is that neither the BoW nor TF-IDF would care about the semantics or context. Which is where Word2Vec is handy and useful.

<h2> Prediction based embedding </h2>
<h3> Word2Vec </h3>

As mentioned above, we would want to have the weightage given to context or the order of the words rather than just a term/word in isolation. In short, we have to consider a context of a word. Word2Vec is a model, which would help us achieve the same using 2 methodologies :

 1. **CBOW (Continuous Bag of Words):** In CBOW model, we would try to ***predict*** the Target word with the help of context word. For ex :- If we see the below sentence, the TARGET word is "language", which is found by using the context words around it. 
 
![enter image description here](https://1.bp.blogspot.com/-gjnDTyhZOmQ/XPDtkX8qtFI/AAAAAAAABtU/lxYFusQZd4UwT9E3aM_mgVj8ldFbYdcawCLcBGAs/s1600/image001.png)
 
The model tries to predict the target word by trying to understand the context of the surrounding words. Consider the same sentence as above, ‘It is a pleasant day’.The model converts this sentence into word pairs in the form *** (contextword, targetword) ***. The user will have to set the window size. If the window for the context word is 2 then the word pairs would look like this: ([it, a], is), ([is, pleasant], a),([a, day], pleasant). With these word pairs, the model tries to predict the target word considered the context words.

2. **Skip-Gram:** Unlike CBOW model, the skip-gram tries to ***predict*** the context words out of the TARGET words. Below is the simple illustration provided by Google on Word2Vec using skip gram with window size of 2 and 3.

Source : https://www.tensorflow.org/tutorials/text/word2vec

![enter image description here](https://www.tensorflow.org/tutorials/text/images/word2vec_skipgram.png)
