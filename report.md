# Q1. Summary
project: 
- I have developed sentiment analysis for a given statement by a user. 
Build:
- I have implemented the NLP using LSTM since it is state-of-the-art technique which captures sequential data with good accuracy.
- I did a lot of preprocessing before sending my input statements to the model(filtering unecessary puntuations, removing stop words, adding POS_tags etc.,).
- I have used embedding layer, bidirectional layer, dropout layer.
- Used model checkpoint.
My approach:
- I have extracted one feature using CNN of Conv1D and given it as additional feature to the testing data to make prediction easier. 
Data:
- My model works best when a sentence contains words that infer emotions of a person like overwhelmed , miserable, contented etc.,
- I have not used any stratified kfold as the datasets are independent of each other.

# Q2. Dataset Description

## Dataset Description
- I have taken my dataset from kaggle. I have three independent datasets, which describes a persons emotions in a sentence and overall feeling associated with that word, seperated by";".
- I have 16000 samples in my training set, 2000 in my testing set, 2000 samples in my validation set. Here labels of my datasets are 6 emotions    joy, love, anger, sadness, surprise, fear. More than 60% of the data have either joy or sadness emotions for all three datasets, other emotions are approximately as follows: 4% surprise, 10% fear, 8% love, 12% anger.
- All of the sentences have detailed description of what people feel in the statements given.
- I am considering 80 timesteps or sequences of words.
- The words such as "as","is","all","and" etc are stop words which will degrade the performance of NLP task at hand.


## AUC Values
As I am developing NLP my dataset is not very much compatible in finding AUC's of the input features.

# Q3. Details
Project:
- As we know that LSTM are the very good units when it comes to time series. Since they have capability of remembering the data for long sequences for long time. Since it has forget gate which rather than having fixed weigths to influence like in RNN's, now it will depend upon the current input data, hidden state of the unit of previous timestamp. 

Model:
- In LSTM used sigmoid for forget_gate update. Using tanh for new information gate as tanh ranges from -1 to 1 it will be able to add or subtract required information form the input that needs to be added to the cell state. Dense layer activation of tanh for LSTM is used. 
- Did pre processing using regular expression. Removed stop words which are not useful in predicting a sentiment from the sequence. Added POS tags to the data to improve performance of recognizing sentiment of the statements given. 
- Padded sequences with zeros at the end to make every sequence same size.
- In LSTM model first I have added embedded layer since it will convert input sequence to a integer vector of given size . The numbers represent the connection between two similar words which is very important in recognizing patterns of input sequence.
- Dropout layer helps in controlling overfitting of model to the given data by making the inputs at certain frequency zero.
- Bidirectional propogates the weight updates past to future and future to past of the sequence. Which will help predict future words of a sequence.
- Used model checkpoint to save the model after every epoch and model while fitting will take best model from current learned weights and saved weights. It will reduce training time and also help getting accurate model.

My approach:
- I thought it would be helpful if we take use of a CNN and try to train it to extract a feature of an sequence and add it to the testing data before sending it through LSTM model. I have noticed slight improvement in the accuracy of test data for many runs. 

Data:
- Since dataset is small and difficult to find dataset with good portion of sentences which complex sentence formation, input needs to be similar to the dataset given. LSTMs will face major challenge Contextual words, phrases, Synonyms, Irony and sarcasm, Ambiguity, Errors in text or speech, Colloquialisms and slang. We need to incorporate much data compared to the data which used in building this model.
