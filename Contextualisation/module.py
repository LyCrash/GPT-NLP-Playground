import yake

# Returns a list of extracted keywords form the text based on the specified language and settings
def extract_keywords(text):

    # Setting frech as the language and n = 10 as the maximum number of keywords
    language = "fr"
    n = 10

    # Create a custom keyword extractor using YAKE with specified settings
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=n, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20, features=None)

    # Extract keywords from the given text using the custom extractor
    keywords = custom_kw_extractor.extract_keywords(text)

    # Return a list containing the extracted keywords (only the first element of each keyword entry)
    return [keyword[0] for keyword in keywords]