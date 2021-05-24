import gensim
from gensim.models import LdaModel
import pandas as pd
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def main():
    # Retrieve CSV data
    csv_data = pd.read_csv('ohsumed-allcats.csv')
    print(csv_data)

    ''' Preprocessing of Data '''
    # Removing rows with Null Values
    updated_data = csv_data.dropna()
    print(updated_data['text'])

    # Removing Special Characters and tabs from the data
    removable_char = str.maketrans("", "", "\t!,.@#$%^&*(){}`\\/[]-+")
    text_data = [word.translate(removable_char) for word in updated_data['text']]

    # Removing characters with length <=2
    # Making caps into lower case
    updated_text_data = [data.lower() for data in text_data if not len(data)<3]

    # Removing digits
    # Removing Stopping Words
    data_list = []
    stopword = set(stopwords.words('english'))
    for data in updated_text_data:
        temp = data.split(" ")
        temp_list = []
        for i in temp:
            if(i not in stopword and i!="" and not i.isdigit() and not len(i)<4):
                temp_list.append(i)
        data_list.append(temp_list)

    # Stemming
    stemmed_list = []
    stemmer = PorterStemmer()
    for temp in data_list:
        stem_temp =[]
        for d in temp:
            stem_temp.append(stemmer.stem(d))
        stemmed_list.append(stem_temp)

    print(stemmed_list[0])

    '''Bag of Words and Dictionary'''
    dict = gensim.corpora.Dictionary(stemmed_list)
    data_corpus = [dict.doc2bow(data) for data in stemmed_list]

    '''gensim LDA model'''
    model = LdaModel(corpus=data_corpus, num_topics = 20, id2word=dict)
    for lda_model in model.print_topics():
        print(lda_model)

    '''Score'''
    c_score = gensim.models.coherencemodel.CoherenceModel(model = model,texts=stemmed_list,dictionary=dict,coherence='c_v')
    print("Coherence Score:", c_score.get_coherence())

if __name__ == "__main__":
    main()