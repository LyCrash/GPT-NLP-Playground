import csv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Method 3 : Semantic Text Matching using sentence embeddings

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


