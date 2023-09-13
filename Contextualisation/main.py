from approches import *
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


#return the response to the prompt using the top_matching_questions method
def chat_with_gpt_embed(title, prompt, context, csv_filename, top_n):
    try:
        
        # Use the top_matching_questions function to get relevant questions and responses
        # select an approach
        
        #relevant_qa_pairs = top_matching_questions_1(prompt, csv_filename, top_n)
        #relevant_qa_pairs = top_matching_questions_2(prompt, csv_filename, top_n)
        relevant_qa_pairs = top_matching_questions_3(prompt, csv_filename, top_n)
        #relevant_qa_pairs = top_matching_questions_4(prompt, csv_filename, top_n)
        
        # select the role, prompt and context
        messages = [
            {"role": "system", "content": title},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": context}
        ]

        # Create a message for each relevant question-response pair
        for qa_pair in relevant_qa_pairs:
            messages.append({"role": "user", "content": qa_pair[0]})
            messages.append({"role": "assistant", "content": qa_pair[1]})

        # Send the messages to ChatGPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Extract the assistant's reply from the response
        reply = response['choices'][0]['message']['content']

        return reply

    except Exception as e:
        return str(e)



#conversation with the chatbot
def main():
    
    print("Bienvenue à BlaBlaCar! \nDites-moi comment je peux vous assister aujourd'hui ?")
    print()

    # Initialize the context as an empty string
    context = ""

    # Give the role to the chatbot
    title = "Tu es un expert en assistance clientèle bienveillant chez BlaBlaCar, la plateforme de covoiturage en ligne"

    # The file used as a source
    csv_filename = "../Datasets/Raw/data.csv"

    # Number of question-answer pairs wanted
    top_n = 3

    while True:
        user_input = input("You: ")

        if user_input.lower() in ('exit', 'quitter'):
            print("Au revoir!")
            break

        # Call the chat_with_gpt function to get the assistant's reply
        assistant_reply = chat_with_gpt_embed(title, user_input, context, csv_filename, top_n)

        print("BlaBlaCar:", assistant_reply)
        print()

        # Update the context with the assistant's reply
        context = assistant_reply



# Calling the main function
if __name__ == "__main__":
    main()