# A text summariser application
A text summariser web app using latent semantic analysis, hosted on AWS EC2 with optional API on AWS Lambda.  
[Web app link](http://13.55.103.147/index)
### API folder
Code to deploy the SLA summariser as a microservice API on AWS Lambda.
### Frontend folder
A web front end implemented in Flask.

### Background - two types of summarisers:  
**Extractive**: A summary is formed by picking the most important sentences in the given text based on a scoring system. Common approaches:

  * Using topic words.  
  * Frequency-driven: how often each word in the sentence appear within and without the sentence.  
  * Latent semantic analysis: An unsupervised method to discover the underlying semantics (topics) of the text.

**Abstractive**: Using advanced analytics techniques such as deep learning (CNN, RNN with LSTM) to generate a new text that conveys the key information in the original [1].

### A summary of the algorithm:
  * The sentences are cleaned and lemmatised (convert words to their root form, i.e. grew to grow).
  * A term-sentence matrix is built with TF-IDF (term frequency inverse document frequency).
  * Apply SVD (singular value decomposition) on the above matrix.
  * Each sentence is given a weight based on how much information they contain for important topics. This approach is based on the sentence length approach proposed by [2].
  * Sentences that are at the top of the text is given a slightly higher weight.
  * Sentences are added to the summary one by one until a set ratio of information is reached.
  
### Reference:  
[1] Allahyari, M, Pouriyeh, S. et al (2017). Text Summarization Techniques: A Brief Survey.  
[2] Steinberger, J, Poesio, M. et al (2007). Two uses of anaphora resolution in summarization.
