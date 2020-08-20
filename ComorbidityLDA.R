## Brief overview of LDA: (from probmods.org)
## 1. For each document, mixture weights over a set of K topics are
##    drawn from a Dirichlet prior
## 2. Then N topics are sampled for the document, one for each
##    word.
##    Each topic is associated with a distribution over words, this
##    distribution is drawn from a Dirichlet prior.
## 3. For each of the N topics drawn for the document, a word is
##    sampled from the corresponding multinomial distribution.
## (I think it's really a categorical distribution, but that's just
## multinomial with n=1)
library(rjags)
library(plyr)
library(tm)
# #==============================
# # Process textmining data
# #==============================
args <- commandArgs(trailingOnly = TRUE)
setwd("C:/Users/Downloads/textmining")
# documents <- readLines("C:/Users/Downloads/textmining/data.txt")
documents <- readLines("C:/Users/Downloads/textmining/test.txt")
## Take our documents and turn it into a Term Document Matrix
## Rows are terms, and columns are documents. Cells are counts of terms in documents.
## Check it out using inspect(mtdm)
## We strip out numbers and punctuation here to make it a bit easier on ourselves
docsToTDM <- function(documents) {
  docs.tm <- Corpus(VectorSource(documents))
  docs.tm <- tm_map(docs.tm, tolower)
  docs.tm <- tm_map(docs.tm, removeNumbers)
  ## docs.tm <- tm_map(docs.tm, removePunctuation)
  docs.tm <- tm_map(docs.tm, function(x) gsub("[[:punct:]]", " ", x))
  docs.tm <- tm_map(docs.tm, stripWhitespace)
  docs.tm <- tm_map(docs.tm, removeWords, stopwords('english'))
  ## We could also filter out non-frequent terms to keep the graph size small:
  ## mtdm <- TermDocumentMatrix(docs.tm)
  ## findFreqTerms(mtdm, 5)
  ## mtdm.sparse <- removeSparseTerms(mtdm, 0.99)
  ## mtdm.sparse
  TermDocumentMatrix(docs.tm)
}
docsToTDM2 <- function(documents) {
  docs.tm <- Corpus(VectorSource(documents))
  TermDocumentMatrix(docs.tm)
}
mtdm <- as.matrix(docsToTDM(documents))
## mtdm <- drop0(mtdm)
## List of all the terms
words <- rownames(mtdm)
## Take in the term document matrix, words, and number of clusters (K)
## Return a JAGS model, which we'll burn in and sample later
genLDA <- function(mtdm, words, K, alpha.Words = 0.1, alpha.Topics = 0.1) {
  ## Here we translate all documents into a numbered matrix, so JAGS can understand it
  ## Each row is a document, it's columns are filled with numbers
  ## Each unique number represents a word in that document
  ## The number of columns is the maximum length of all documents
  ## Unused columns are filled with "NA"
  word <- do.call(rbind.fill.matrix,
                  lapply(1:ncol(mtdm), function(i) t(rep(1:length(mtdm[,i]), mtdm[,i]))))
  N <- ncol(mtdm)                 #Number of documents
  Nwords <- length(words)         #Number of terms
  alphaTopics <- rep(alpha.Topics, K)      #Hyperprior on topics
  alphaWords <- rep(alpha.Words, Nwords)  #Hyperprior on words
  ## These hyperpriors are set such that we can give weights such as (1,0,0) to topics.
  ## If we had 3 topics and used an alpha of (100, 100, 100), we'd
  ## only expect relatively even mixture weights on the
  ## topics. This isn't what we generally want. We'd like documents
  ## to be able to belong to mostly one topic.
  ## For each word in a document, we sample a topic
  wordtopic <- matrix(NA, nrow(word), ncol(word))
  ## Length of documents needed for indexing in JAGS
  doclengths <- rowSums(!is.na(word))
  ## How much we believe each document belongs to each of K topics
  topicdist <- matrix(NA, N, K)
  ## How much we believe each word belongs to each of K topics
  topicwords <- matrix(NA, K, Nwords)
  ## All the parameters to be passed to JAGS
  dataList <- list(alphaTopics = alphaTopics,
                   alphaWords = alphaWords,
                   topicdist = topicdist,
                   wordtopic = wordtopic,
                   word = word,
                   Ndocs = N,
                   Ktopics = K,
                   length = doclengths,
                   Nwords = Nwords,
                   worddist = topicwords)
  
  jags.model('model.jag',
             data = dataList,
             n.chains = 5,
             n.adapt = 100)
}
## Take a look at how topics are distinguished
## For each word, show it's association with topics
wordsToClusters <- function(jags, words, n.iter = 100) {
  sampleTW <- jags.samples(jags,
                           c('worddist'),
                           n.iter)$worddist
  colnames(sampleTW) <- words
  sTW <- summary(sampleTW, FUN = mean)$stat
  sTW[,order(colSums(sTW))]
  t(sweep(sTW,2,colSums(sTW), '/'))
}
## Lets assign topics to the documents
## We sample from "topicdist" and pick the topic with highest weight
labelDocuments <- function(jags, n.iter = 1000) {
  topicdist.samp <- jags.samples(jags,
                                 c('topicdist'),
                                 n.iter)
  marginal.weights <- summary(topicdist.samp$topicdist, FUN = mean)$stat
  best.topic <- apply(marginal.weights, 1, which.max)
  best.topic
}
##############################
genLDA <- function(mtdm, words, K, alpha.Words = 0.1, alpha.Topics = 0.1) {
  ## Here we translate all documents into a numbered matrix, so JAGS can understand it
  ## Each row is a document, it's columns are filled with numbers
  ## Each unique number represents a word in that document
  ## The number of columns is the maximum length of all documents
  ## Unused columns are filled with "NA"
  word <- do.call(rbind.fill.matrix,
                  lapply(1:ncol(mtdm), function(i) t(rep(1:length(mtdm[,i]), mtdm[,i]))))
  wordCount <- t(mtdm)
  N <- ncol(mtdm)                 #Number of documents
  Nwords <- length(words)         #Number of terms
  alphaTopics <- rep(alpha.Topics, K)      #Hyperprior on topics
  alphaWords <- rep(alpha.Words, Nwords)  #Hyperprior on words
  ## These hyperpriors are set such that we can give weights such as (1,0,0) to topics.
  ## If we had 3 topics and used an alpha of (100, 100, 100), we'd
  ## only expect relatively even mixture weights on the
  ## topics. This isn't what we generally want. We'd like documents
  ## to be able to belong to mostly one topic.
  ## For each word in a document, we sample a topic
  wordtopic <- matrix(NA, nrow(word), Nwords)
  ## Length of documents needed for indexing in JAGS
  doclengths <- rowSums(!is.na(word))
  ## How much we believe each document belongs to each of K topics
  topicdist <- matrix(NA, N, K)
  ## How much we believe each word belongs to each of K topics
  worddist <- matrix(NA, K, Nwords)
  ## All the parameters to be passed to JAGS
  dataList <- list(alphaTopics = alphaTopics,
                   alphaWords = alphaWords,
                   topicdist = topicdist,
                   wordtopic = wordtopic,
                   wordCount = wordCount,
                   Ndocs = N,
                   Ktopics = K,
                   Nwords = Nwords,
                   worddist = worddist)
  
  jags.model('model.jag',
             data = dataList,
             n.chains = 5,
             n.adapt = 100)
}

genLDA2 <- function(mtdm, words, K, E, alpha.Words = 0.1, alpha.Topics = 0.1) {
  ## Here we translate all documents into a numbered matrix, so JAGS can understand it
  ## Each row is a document, it's columns are filled with numbers
  ## Each unique number represents a word in that document
  ## The number of columns is the maximum length of all documents
  ## Unused columns are filled with "NA"
  word <- do.call(rbind.fill.matrix,
                  lapply(1:ncol(mtdm), function(i) t(rep(1:length(mtdm[,i]), mtdm[,i]))))
  wordCount <- t(mtdm)
  N <- ncol(mtdm)                 #Number of documents
  Nwords <- length(words)         #Number of terms
  alphaTopics <- rep(alpha.Topics, K)      #Hyperprior on topics
  alphaWords <- rep(alpha.Words, Nwords)  #Hyperprior on words
  ## These hyperpriors are set such that we can give weights such as (1,0,0) to topics.
  ## If we had 3 topics and used an alpha of (100, 100, 100), we'd
  ## only expect relatively even mixture weights on the
  ## topics. This isn't what we generally want. We'd like documents
  ## to be able to belong to mostly one topic.
  ## For each word in a document, we sample a topic
  wordtopic <- matrix(NA, nrow(word), Nwords)
  ## Length of documents needed for indexing in JAGS
  doclengths <- rowSums(!is.na(word))
  ## How much we believe each document belongs to each of K topics
  topicdist <- matrix(NA, N, K)
  ## How much we believe each word belongs to each of K topics
  worddist <- matrix(NA, K, Nwords)
  ## Initialize E
  E <- matrix(NA, Nwords, N)
  ## Initialize gamma
  gamma <- t(rep(NA, N))
  ## All the parameters to be passed to JAGS
  dataList <- list(alphaTopics = alphaTopics,
                   alphaWords = alphaWords,
                   topicdist = topicdist,
                   wordtopic = wordtopic,
                   wordCount = wordCount,
                   Ndocs = N,
                   Ktopics = K,
                   Nwords = Nwords,
                   worddist = worddist,
                   E = E,
                   gamma = gamma)
  
  jags.model('model2.jag',
             data = dataList,
             n.chains = 5,
             n.adapt = 100)
}
##############################
## Burn our chain in.
## Hopefully.
## Ideally we'd do some plots showing mixing.
## Word of warning: The number of updates and chains I use is really arbitrary, and not
## carefully selected.)
jags <- genLDA(mtdm, words, 15)
update(jags, 5000)
jags2 <- genLDA2(mtdm, words, 15, E)
update(jags2, 5000)
# split(documents, labelDocuments(jags))
sampleTW2 <- jags.samples(jags2,
                          c('worddist'),
                          100)$worddist
colnames(sampleTW2) <- words
sTW2 <- summary(sampleTW2, FUN = mean)$stat
write.table(sTW2, file="sTW2.txt", row.names=FALSE, col.names=TRUE)
library(wordcloud)
genWordCloud <- function(sampleTW, words, columnNumber,...) {
  freq <- t(summary(sampleTW, FUN = mean)$stat)[,columnNumber]
  wordcloud(words[order(freq)],freq[order(freq)],...)
}
par(mfrow=c(1,6))
for(i in 1:6)
  genWordCloud(sampleTW2, words, i, min.freq = 0.01)