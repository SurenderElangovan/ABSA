import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import _pickle as cPickle
import os
import html

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def construct_data(review):
    data = {
              "ugcid": [213584],
              "userid":[1027],
              "ugc":review,
              "aspectid":["386,387,388,389,390,000"],
              "categoryid":[61]
            }
    df = pd.DataFrame(data)
    return df
# Reading Text File
def start_file_read(filepath):
    data_txt = pd.read_csv(filepath, delimiter="\t",engine='python')
    # Taking not null columns from aspectid column
    print("Opening file")
    data_txt = data_txt[data_txt['aspectid'].notnull()]
    return data_txt

# Text Preprocessing Function
def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Lemmatization, Stemming, and Removing Stopwords
def data_preprocessing(data_txt):
    print("Pre processing.")
    stop = stopwords.words('english')
    wl = WordNetLemmatizer()
    data_txt['ugc'] = data_txt['ugc'].apply(lambda x: " ".join([wl.lemmatize(word) for word in x.split() if word not in stop]))

    # Cleaning text: lowercasing, removing punctuation, and extra whitespace
    data_txt['ugc'] = data_txt['ugc'].str.lower().str.replace("'", '').str.replace('[^\w\s]', ' ').str.replace(" \d+", " ").str.replace(' +', ' ').str.strip()

    # Converting ugc into string
    data_txt["ugc"] = data_txt["ugc"].astype(str)
    return data_txt

# Academic Sentiment Prediction
def predict_Academic_sentiment(data_txt):
    print("Scanning Acadamics Sentiment")
    data_txt['Academic Sentiment'] = 0
    t_academic = []
    cv_academic = cPickle.load(open('Pickle files/TF_Vectorizer_Academic386.pickle', 'rb'))
    for i in range(len(data_txt)):
        nltk_tokens = sent_tokenize(data_txt.iloc[i, 2])
        for j in nltk_tokens:
            sentence = cv_academic.transform(list([j])).toarray()
            academic_classifier = cPickle.load(open('Pickle files/RF_Academic_Sentiment_Model386.pickle', 'rb'))
            Yacademic = academic_classifier.predict(sentence)
            for k in range(len(sentence)):
                data_txt.iloc[i, 5] = Yacademic[k]
                t_academic.append(Yacademic[k])
    return data_txt

# Campus Sentiment Prediction
def predict_campus_sentiment(data_txt):
    print("Scanning Campus Sentiment")
    data_txt['Campus Sentiment'] = 0
    t_campus = []
    cv_campus = cPickle.load(open('Pickle files/TF_Vectorizer_Campus387.pickle', 'rb'))
    for i in range(len(data_txt)):
        nltk_tokens = sent_tokenize(data_txt.iloc[i, 2])
        for j in nltk_tokens:
            sentence = cv_campus.transform(list([j])).toarray()
            campus_classifier = cPickle.load(open('Pickle files/RF_Campus_Sentiment_Model387.pickle', 'rb'))
            Ycampus = campus_classifier.predict(sentence)
            for k in range(len(sentence)):
                data_txt.iloc[i, 6] = Ycampus[k]
                t_campus.append(Ycampus[k])
    return data_txt

# Fee Sentiment Prediction
def predict_fee_sentiment(data_txt):
    print("Scanning Fees Sentiment")
    data_txt['Fee Sentiment'] = 0
    t_fee = []
    cv_fee = cPickle.load(open('Pickle files/TF_Vectorizer_Fee388.pickle', 'rb'))
    for i in range(len(data_txt)):
        nltk_tokens = sent_tokenize(data_txt.iloc[i, 2])
        for j in nltk_tokens:
            sentence = cv_fee.transform(list([j])).toarray()
            fee_classifier = cPickle.load(open('Pickle files/RF_Fee_Sentiment_Model388.pickle', 'rb'))
            Yfee = fee_classifier.predict(sentence)
            for k in range(len(sentence)):
                data_txt.iloc[i, 7] = Yfee[k]
                t_fee.append(Yfee[k])
    return data_txt

# Online learning Sentiment Prediction
def predict_online_learning_sentiment(data_txt):
    print("Scanning Online-Learning Sentiment")
    data_txt['Online learning Sentiment'] = 0
    t_online = []
    cv_online = cPickle.load(open('Pickle files/TF_Vectorizer_Online_learning389.pickle', 'rb'))
    for i in range(len(data_txt)):
        nltk_tokens = sent_tokenize(data_txt.iloc[i, 2])
        for j in nltk_tokens:
            sentence = cv_online.transform(list([j])).toarray()
            online_learning_classifier = cPickle.load(open('Pickle files/RF_Online_learning_Sentiment_Model389.pickle', 'rb'))
            Yonline = online_learning_classifier.predict(sentence)
            for k in range(len(sentence)):
                data_txt.iloc[i, 8] = Yonline[k]
                t_online.append(Yonline[k])
    return data_txt

# Placements Sentiment Prediction
def predict_placement_sentiment(data_txt):
    print("Scanning Placements Sentiments")
    data_txt['Placements Sentiment'] = 0

    t_placements = []
    cv_placements = cPickle.load(open('Pickle files/TF_Vectorizer_Placements390.pickle', 'rb'))
    for i in range(len(data_txt)):
        nltk_tokens = sent_tokenize(data_txt.iloc[i, 2])
        for j in nltk_tokens:
            sentence = cv_placements.transform(list([j])).toarray()
            placements_classifier = cPickle.load(open('Pickle files/RF_Placements_Sentiment_Model390.pickle', 'rb'))
            Yplacements = placements_classifier.predict(sentence)
            for k in range(len(sentence)):
                data_txt.iloc[i, 9] = Yplacements[k]
                t_placements.append(Yplacements[k])
    return data_txt

# Defining sentiment function using TextBlob
def sentiment(x):
    polarity_value = TextBlob(x).sentiment.polarity
    if polarity_value >= 0.05:
        return "Positive"
    elif polarity_value <= -0.05:
        return "Negative"
    else:
        return "Neutral"
    
#Predicting the overall sentiment    
def predict_overall_sentiment(data_txt):
    print("overall")
    data_txt['overallsentiment'] = data_txt['ugc'].apply(sentiment)
    # Creating a copy of the given dataframe
    data_final = data_txt.copy(deep=True)
    return data_final

def add_emoticon(value):
    if value == "Positive":
        return f"{value} 😊"  # Smiling emoticon for positive values
    elif value == "Negative":
        return f"{value} 😞"  # Sad emoticon for negative values
    elif value == "Neutral":
        return f"{value} 😐" # emoticon for neutral values
    else:
        return f"{value} ❌"  # X icon for zero values or others

def post_processing(data_final,returndata,filepath):
    print("post processing")
    #filename="Sentiment_Analysis_Final_Output_text_RF_Alternate.xlsx"
    # Dropping unnecessary columns
    #data_final.drop(columns=['ugc', 'categoryid'], inplace=True)
    data_final.drop(columns=['categoryid','aspectid'], inplace=True)
    if returndata:
        data_final.to_excel(filepath)
        return os.path.basename(filepath)
    else:
        data_final=data_final.drop(['ugcid', 'userid'], axis=1)
        data_final.rename(columns = {'ugc':'Review'}, inplace = True)
        sentiment_columns = ['Academic Sentiment', 'Campus Sentiment', 'Fee Sentiment',
                     'Online learning Sentiment', 'Placements Sentiment','overallsentiment']

        for column in sentiment_columns:
            data_final[column] = data_final[column].apply(add_emoticon)
        data_final.applymap(lambda x: html.unescape(x) if pd.notnull(x) else x)
        return data_final.to_html(index=False)
    #data_copy_2.to_excel("Education_Output_Tool_Format_Updated.xlsx")
    
#Driver code
def main(review,filepath):
    result={"file":"","review":""}
    for key in result.keys():
        returndata = False
        if key == "file":
            if not filepath:
                continue
            returndata = True
            data_txt=start_file_read(filepath)
            filename, file_extension = os.path.splitext(os.path.basename(filepath))
            new_file_path = os.path.join(os.path.dirname(filepath), filename+"_output" +".xlsx")
            
        elif key == "review":
            if not review:
                continue
            data_txt=construct_data(review)
            new_file_path = None
            
        data_txt=data_preprocessing(data_txt)
        data_txt=predict_Academic_sentiment(data_txt)
        data_txt=predict_campus_sentiment(data_txt)
        data_txt=predict_fee_sentiment(data_txt)
        data_txt=predict_online_learning_sentiment(data_txt)
        data_txt=predict_placement_sentiment(data_txt)
        data_txt=predict_overall_sentiment(data_txt)
        data_txt=post_processing(data_txt,returndata,new_file_path)
        result[key]=data_txt
    return result

if __name__=="__main__":
    main()
