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



## Links to Relevant Notebooks
* [Preprocessing Text and Stop Words](https://github.com/JDHazard/tolkien_nlp_capstone/blob/master/preprocessing_stopwords_plotly.ipynb)
* [Topic Modelling on the Complete LotR Text](https://github.com/JDHazard/tolkien_nlp_capstone/blob/master/topic_modeling_LotR_complete.ipynb)
* [Word2Vec on the Complete LotR Text](https://github.com/JDHazard/tolkien_nlp_capstone/blob/master/word2vec_lotr_complete.ipynb)
* [Topic Modelling on Each Book of the Trilogy]()


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
