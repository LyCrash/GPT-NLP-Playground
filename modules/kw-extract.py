from rake_nltk import Rake
import nltk
import yake
import pandas as pd

text="Ceci est un simple exemple qui contient des mots-clès à extraire"

#The higher the score, the more relevant the keyword is.
def kw_extract_rake(phrase):
    r = Rake()  # Initialize Rake-NLTK
    r.extract_keywords_from_text(phrase)
    # To get keyword phrases ranked highest to lowest.
    keywords = r.get_ranked_phrases()

    # To get keyword phrases ranked highest to lowest with scores.
    #keywords = r.get_ranked_phrases_with_scores()
    return keywords

def kw_extract_rake_fr(phrase): 
    r = Rake(language="french")
    r.extract_keywords_from_text(phrase)
    keywords = r.get_ranked_phrases()
    return keywords

#The higher the score, the more relevant the keyword is.
def kw_extract_yake(phrase):
    ext = yake.KeywordExtractor()
    keywords = ext.extract_keywords(phrase)
        # Get the keywords only from the tuple result (keyword,score)
    keywords_only = [keyword for keyword, _ in keywords]
    return keywords_only

def kw_extract_yake_fr(phrase):
    language = "fr"
    max_ngram_size = 1
    deduplication_threshold = 0.7
    deduplication_algo = 'seqm'
    numOfKeywords = 5
    stopwords = set(nltk.corpus.stopwords.words('french'))

    ext = yake.KeywordExtractor(
        lan = language,
        n = max_ngram_size,
        dedupFunc = deduplication_algo,
        dedupLim = deduplication_threshold,
        # use_stopwords = True available on version 0.5.0
        stopwords = stopwords
    )
    keywords = ext.extract_keywords(phrase)

        # Get the keywords only from the tuple result (keyword,score)
    keywords_only = [keyword for keyword, _ in keywords]
    return keywords_only    


# returns the index of a line in the "data.csv" file
def find_index(line):
    # Read the CSV file
    data = pd.read_csv('..\Datasets\Raw\data.csv')
    # Find the index (row number) of the given prompt
    index = data[ (data['Question'] == line) | (data['Reponse'] == line) ].index
    return index.item() if index.any() else None

# print("Sentence= ",text)
# print("\n_______________ Using RAKE method\nKeywords = ",kw_extract_rake_fr(text))
# print("\n_______________ Using YAKE method\nKeywords = ",kw_extract_yake_fr(text))
# print(find_index(text),f": {text}")