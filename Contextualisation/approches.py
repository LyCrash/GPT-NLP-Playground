import csv
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from module import extract_keywords

# Approach 1 : Semantic Text Matching by compaing strings - jaccard similarity
# returns the top_n most similar questions/responses to the input_prompt by comparing strings

def top_matching_questions_1(input_prompt, csv_filename, top_n=3):

    result = []

    # Extract the words from the input prompt
    input_words = input_prompt.split()

    with open(csv_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            question = row['Question']
            response = row['Reponse']

            # Extract the words from the question
            question_words = question.split()

            # Calculate the similarity between the input prompt and each question

              # '&' represents the intersection of two sets of words
              # '|' represents the union of two sets of words
              # 'len' represents the number of elements
              # The formula calculates the number of common words divided by the number of total words

            similarity = len(set(input_words) & set(question_words)) / len(set(input_words) | set(question_words))

            # Add the tuple (question, response, similarity) to the result array
            result.append((question, response, similarity))


    # Sort the results based on similarity in descending order
    result.sort(key=lambda x: x[2], reverse=True)

    # Return the top questions
    return result[:top_n]





#Approach 2 : Semantic Text Matching by comparing substrings
# returns the top_n most similar questions/responses to the input_prompt by comparing strings

def top_matching_questions_2(input_prompt, csv_filename, top_n=3):
    result = []
    with open(csv_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            question = row['Question']
            response = row['Reponse']

            # calculate the similarity between the imput and the question
            similarity = SequenceMatcher(None, input_prompt, question).ratio()

            #add the tuple (question, response, similarity) to the result array
            result.append((question, response, similarity))

    # Sort the results based on similarity in descending order
    result.sort(key=lambda x: x[2], reverse=True)

    # Return the top questions
    return result[:top_n]





# Approach 3 : Semantic Text Matching using sentence embeddings
# returns the top_n most similar questions/responses to the input_prompt using sentence embeddings
def top_matching_questions_3(input_prompt, csv_filename, top_n):
    result = []

    # Use a pre-trained sentence embeddings model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    with open(csv_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            question = row['Question']
            response = row['Reponse']

            # Convert sentences into embeddings
            input_embed = model.encode(input_prompt, convert_to_tensor=True)
            question_embed = model.encode(question, convert_to_tensor=True)

            # Calculate cosine similarity between embeddings
            similarity = cosine_similarity(input_embed.reshape(1, -1), question_embed.reshape(1, -1))[0][0]

            # Add the tuple (question, response, similarity) to the results
            result.append((question, response, similarity))

    # Sort the results based on similarity in descending order
    result.sort(key=lambda x: x[2], reverse=True)

    # Return the top questions
    return result[:top_n]





#Approach 4 : TF-IDF Semantic Text Matching using keywords
# returns the top_n most similar questions/responses to the input_prompt using tf-idf semantic search and based on keywords extraction

def top_matching_questions_4(input_prompt, csv_file, top_n):

    result = []

    # Load data from the CSV file
    with open(csv_file, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            question = row[0]
            response = row[1]
            result.append((question, response))

    # Extract keywords from the input prompt using the 'extract_keywords' function
    input_keywords = extract_keywords(input_prompt)

    # Extract keywords from questions in the CSV using the 'extract_keywords' function
    question_keywords_list = [extract_keywords(question) for question, _ in result]

    # Use CountVectorizer to create keyword vectors for questions
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    question_vectors = vectorizer.fit_transform(question_keywords_list)
    input_vector = vectorizer.transform([input_keywords])

    # Calculate cosine similarity between the input vector and question vectors
    similarities = cosine_similarity(input_vector, question_vectors)[0]

    # Sort results in descending order of similarity
    sorted_results_indices = similarities.argsort()[::-1]
    sorted_results = [(result[idx][0], result[idx][1], similarities[idx]) for idx in sorted_results_indices]

    return sorted_results[:top_n]