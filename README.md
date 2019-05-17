# Arresting Strangeness: NLP Analysis of The Lord of the Rings

## Table of Contents

* [Problem Statement](#user-content-problem-statement)
* [Executive Summary](#user-content-executive-summary)
* [Links to Relevant Notebooks](#user-content-links-to-relevant-notebooks)
* [Opportunities for Further Study](#user-content-opportunities-for-further-study)
* [References](#user-content-references)

## Problem Statement

There are copious volumes of literary criticism about and surrounding J.R.R. Tolkien's 'The Lord of the Rings'.  Close readings, historical analysis, queer readings, deconstruction, and many more methods have been deployed to elucidate Tolkien's work. Natural Language Processing, Topic Modeling, and Word2Vec analysis are relatively new technologies. What are the results of using these technologies to analyze Tolkien's immortal tome? Can we answer specific questions regarding the text? Is the analysis deeper and more nuanced than the traditional methods?
My goal is to use NLP as a method of literary criticism and, much like the X variables in an unsupervised learning model, observe and report trends and relationships in the text. 

## Executive Summary

There is unprecedented access to the written word. The impossibility of reading and understanding even a small percentage of it is daunting. Machine learning algorithms such as Topic Modeling and Word2Vec have proven useful in classifying general topics in non-fiction and scientific documents. Can these methods and techniques be used to illuminate one of the greatest literary achievements in the English language? Let's find out. 

Corpus: 
['The Lord of the Rings'] (https://en.wikipedia.org/wiki/The_Lord_of_the_Rings) Written by English author and scholar [J.R.R. Tolkien](https://en.wikipedia.org/wiki/J._R._R._Tolkien) in stages between 1937 and 1949. It is widely considered the germinal work of epic high fantasy. 
Split into three parts by the publisher, the majority of our textual analysis is on the complete work. 

Preprocessing: 
For us to fit our text, or corpus, to any model and expect meaningful results, we must first go through a few stages of preprocessing. Removing punctuation, reducing each word to its root or lemma, and eliminating stop words may seem crude and primitive at the outset. Unfortunately, our computers' programming cannot understand the raw text. They must be taught to 'read' it by assigning words to numerical values. Preprocessing is the first step in this process. 
Stop Words:
Our models are sensitive to the removal of words from the corpus because they depend on words counts and the relationships between them to interpret the text. Stop words are the most commonly used words, mostly, and prepositions, that interfere with an algorithm's predictions. Why? Because of their document frequency, the model thinks they're essential. 

Topic Modelling: 
Is a process of discovering abstract topics regarding a specific text. For example, a document that frequently mentions the words 'kitty', 'scratch', 'kitten' and 'meow' is most likely about 'cats'. While there may be words like 'food', 'poop', and 'tail', these words could also belong to a document on 'dogs'. In machine learning, it is a form of dimensionality reduction, unsupervised learning, and tagging. Topic modeling has proven useful in text classification by grouping words together and using them as a feature in recommender systems. Also, find out what's 'trending' in groups of documents has been used in social media. 

Topic Modelling & Fiction: 
I chose to use both the latent Dirichlet allocation (LDA) and Non-negative matrix factorization (NMF or NNMF) models to use on The Lord of the Rings. Although fitting and interpreting the topics from these models proved challenging, our results, were mixed. At best, we got an approximation of what a topic might be about, but it took my knowledge of the subject to figure out what that topic may be. At worst, we got a sort of tone poem about a particular subject which could be useful to someone looking for a new critical approach to the text. 

Example topics from LDA and NMF:
LDA Topic #2: 'merry, city, rider, Legolas, Rohan, wish, save, Boromir, guess, followed, caught, used, running, deed, ship, try, strider, tongue, shagrat, nearly' -- Is this a topic about saving Boromir? Hard to say, mainly because our model is unsupervised. That is, we don't have access to a target variable to measure our results. 
NMF Topic #17: 'far, good, shadow, think, ever, side, passed, though, water, Gollum' -- Gollum and water are indeed connected in the story. Is my model alluding to that topic?
Ultimately, the LDA and NMF models assume that a writer is using ceratin distributions of words in a particular order to create meaning. While that may be the case for non-fiction, a good fiction writer is trying to figure out smart ways of breaking those rules. 

Word2Vec:
This process is best summed up by this quote: 'The purpose and usefulness of Word2vec are to group the vectors of similar words together in vectorspace. That is, it detects similarities mathematically. Word2vec creates vectors that are distributed numerical representations of word features, features such as the context of individual words. It does so without human intervention.

Given enough data, usage and contexts, Word2vec can make highly accurate guesses about a word’s meaning based on past appearances. Those guesses can be used to establish a word’s association with other words (e.g. “man” is to “boy” what “woman” is to “girl”), or cluster documents and classify them by topic. Those clusters can form the basis of search, sentiment analysis and recommendations in such diverse fields as scientific research, legal discovery, e-commerce and customer relationship management.' (https://skymind.ai/wiki/word2vec)

Word@Vec & Fiction: 
I fit a bag of words (CBOW) model and a skip gram model. Both the bag of words model and the skip-gram models provided some interesting insights into the text. For example, when the most similar cosine vectors or similar words for shire were not 'hobbits' or 'halflings' but 'war' and 'fear'. Similarly, the character Tom Bombadil was met with such weird pairings as 'maggot' and 'shouting'. 

My recommendations moving forward are to examine the fundamental assumptions these models make and how they can be leveraged to suit fiction better. 


## Links to Relevant Notebooks
* [Preprocessing Text and Stop Words](https://github.com/JDHazard/tolkien_nlp_capstone/blob/master/preprocessing_stopwords_plotly.ipynb)
* [Topic Modelling on the Complete LotR Text](https://github.com/JDHazard/tolkien_nlp_capstone/blob/master/topic_modeling_LotR_complete.ipynb)
* [Word2Vec on the Complete LotR Text](https://github.com/JDHazard/tolkien_nlp_capstone/blob/master/word2vec_lotr_complete.ipynb)
* [Topic Modelling on Each Book of the Trilogy](https://github.com/JDHazard/tolkien_nlp_capstone/tree/master/topic_modelling_books_trilogy)


## Opportunities for Further Study

* Integration of 'The Hobbit' as primary source material
    - Can a Doc2Vec model be trained to recognize text from 'The Hobbit' against text from 'The Lord of the Rings'
    - Generate new topics and new word vectors
* Integration of Tolkien's non-fiction writings, particularly ['On Fairy Stories'](http://heritagepodcast.com/wp-content/uploads/Tolkien-On-Fairy-Stories-subcreation.pdf)
* Plotting Topics across books or chapters
    - Use the linear progression of the text as a time/series proxy
* Further research into topic modelling 
    - Get our topics to 'converge'
    - Is there a optimal topic model for fiction?

## References
* [*The Lord of the Rings Complete Text*](https://archive.org/details/TheLordOfTheRing1TheFellowshipOfTheRing)
* [Spooky NLP and Topic Modelling tutorial](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)
* [Topic Modeling and Latent Dirichlet Allocation (LDA) in Python](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)
