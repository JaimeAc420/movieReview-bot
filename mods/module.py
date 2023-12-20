import warnings
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report,confusion_matrix
import unicodedata


warnings.filterwarnings("ignore")

import pandas as pd

#Read data from Google Sheet: experimentos.analitica.datos - EncuestaCineColombiano_Respuestas
df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQC3CXrmRk6mpK9-DrpO--faGVT_KsR8rj-AToUfFlsbKNnUB2wVslmNYiFT1pv80Z5gp76tgSqp1aN/pub?gid=1802142849&single=true&output=tsv", sep="\t")

df.columns = ['A','B','C','D','E']

#Transform dataset: Opinion-Type
good_df = df[['C']]
good_df['Opinion'] = "POSITIVE"

bad_df = df[['D']]
bad_df.columns = ['C']
bad_df['Opinion'] = "NEGATIVE"

df_op = pd.concat([good_df,bad_df])
df_op.columns = ['Opinion','Type']

df_op.groupby(['Type']).count()

nltk.download('stopwords')
stemmer = SnowballStemmer('spanish')
nltk.download('punkt')

stop_words = set(stopwords.words('spanish'))
stop_words = stop_words.union(set(['pelicul', 'colombian', 'cin', 'me', 'le', 'da', 'mi', 'su', 'ha', 'he', 'ya', 'un', 'una', 'es','del', 'las', 'los', 'en', 'que', 'y', 'la','de']))

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def fast_preproc(text):
  text = text.lower()
  text = ''.join(c for c in text if not c.isdigit())
  text = ''.join(c for c in text if c not in punctuation)
  text = remove_accents(text)
  words = word_tokenize(text)
  words = [stemmer.stem(word) for word in words]
  words = [word for word in words if not word in stop_words]
  try:
    text = " ".join(str(word) for word in words)
  except Exception as e:
    print(e)
    pass
  return text

df_op['Opinion'] = df_op['Opinion'].astype(str)

df_op = df_op.assign(
    TextPreproc=lambda df: df_op.Opinion.apply(fast_preproc)
)

X = df_op['TextPreproc']
Y = df_op['Type']

vec = TfidfVectorizer(max_df=0.5)

#Tokenize and build vocabulary
vec.fit(X)


#Encode documents
trans_text_train = vec.transform(X)

#Print Document-Term Matrix
df = pd.DataFrame(trans_text_train.toarray(), columns=vec.get_feature_names_out())

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.1)


classifier = LinearSVC()

classifier.fit(X_train, Y_train)
#y_pred = classifier.predict(X_test)

def predict_opinion(text):
    Xt_new = [fast_preproc(str(text))]

    trans_new_doc = vec.transform(Xt_new) #Use same TfIdfVectorizer

    return str(classifier.predict(trans_new_doc))




