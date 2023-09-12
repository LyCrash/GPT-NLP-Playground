import openai, os, sys
from dotenv import load_dotenv

sys.path.append('D:\\2CS_esi\SPE\GPT-NLP-Playground\Contextualisation')
from approches import top_matching_questions_3 

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")   # Ornidex api key

# davinci:ft-personal-2023-08-22-12-58-39 : a fine-tuned model trained on an english dataset of 10 samples "data10_prepared.jsonl"
# davinci:ft-personal-2023-08-22-15-13-42 : a fine-tuned model trained on a french dataset of 50 samples "data_pcp_prepared.jsonl"
def chat_with_mydavinci(prompt):
    try:
         # Call the OpenAI API using the /v1/completions endpoint
        response = openai.Completion.create(
            engine="davinci:ft-personal-2023-08-22-12-58-39",  # You can change the model if needed
            prompt= prompt,
            max_tokens= 100
        )

        # Extract the assistant's reply from the response
        reply = response['choices'][0].text
        return reply
    except Exception as e:
        return str(e)


# ft:gpt-3.5-turbo-0613:personal::7surZ172 : a fine-tuned model trained on a french dataset of 50 samples "data_ccf.jsonl"
def chat_with_mygpt_brut(prompt):
    try:
        systemRole = "Tu es un expert en assistance clientèle bienveillant chez BlaBlaCar, la plateforme de covoiturage en ligne"
        # Call the OpenAI API using the /v1/chat/completions endpoint
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::7surZ172",  # to be changed when needed
            messages=[
                {"role": "system", "content": systemRole},
                {"role": "user", "content": prompt},
            ]
        )
        # Extract the assistant's reply from the response
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        return str(e)


def chat_with_mygpt_ctx_asc(prompt, context):
    try:
        systemRole = "Tu es un expert en assistance clientèle bienveillant chez BlaBlaCar, la plateforme de covoiturage en ligne"
        # Call the OpenAI API using the /v1/chat/completions endpoint
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::7surZ172",  # to be changed when needed
            messages=[
                {"role": "system", "content": systemRole},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": context}
            ]
        )
        # Extract the assistant's reply from the response
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        return str(e)


def chat_with_mygpt_ctx_desc(prompt, csv_filename, top_n):
    try:
        systemRole = "Tu es un expert en assistance clientèle bienveillant chez BlaBlaCar, la plateforme de covoiturage en ligne"
        messages = [
                {"role": "system", "content": systemRole},
                {"role": "user", "content": prompt},
            ]
        
        # Prepare the context using PE-Embeddings algorithms # METHOD3

            # Get the (top_n) relevant questions and responses
        relevant_qa_pairs = top_matching_questions_3(prompt, csv_filename, top_n)
            # Create a message for each relevant question-response pair
        for qa_pair in relevant_qa_pairs:
            messages.append({"role": "user", "content": qa_pair[0]})
            messages.append({"role": "assistant", "content": qa_pair[1]})



        # Call the OpenAI API using the /v1/chat/completions endpoint
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::7surZ172",  # to be changed when needed
            messages=messages
        )

        # Extract the assistant's reply from the response
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        return str(e)


def chat_with_mygpt_hybrid(prompt, context, csv_filename, top_n):
    try:
        systemRole = "Tu es un expert en assistance clientèle bienveillant chez BlaBlaCar, la plateforme de covoiturage en ligne"
        messages = [
                {"role": "system", "content": systemRole},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": context}
            ]
        # Prepare the context using PE-Embeddings algorithms # METHOD3
            # Get the (top_n) relevant questions and responses
        relevant_qa_pairs = top_matching_questions_3(prompt, csv_filename, top_n)
            # Create a message for each relevant question-response pair
        for qa_pair in relevant_qa_pairs:
            messages.append({"role": "user", "content": qa_pair[0]})
            messages.append({"role": "assistant", "content": qa_pair[1]})
        # Call the OpenAI API using the /v1/chat/completions endpoint
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal::7surZ172",  # to be changed when needed
            messages=messages
        )
        # Extract the assistant's reply from the response
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        return str(e)



def main():
    print("Welcome to our fine-tuned BlaGPT!")
    
    context=""
    dataset = "..\Datasets\Raw\data.csv"
    top_n = 1

    while True:
        user_input = input("You: ")

        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break

        # Call the chat_with_gpt function to get the assistant's reply

        assistant_reply = chat_with_mydavinci(user_input)
        #assistant_reply = chat_with_mygpt_brut(user_input)
        #assistant_reply = chat_with_mygpt_ctx_asc(user_input, context)
        #assistant_reply = chat_with_mygpt_ctx_desc(user_input, dataset, top_n)
        #assistant_reply = chat_with_mygpt_hybrid(user_input, context, dataset, top_n)

        print("BlaGPT:", assistant_reply)
        
        context = assistant_reply


if __name__ == "__main__":
    main()

