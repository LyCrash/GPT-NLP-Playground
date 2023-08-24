import openai, os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")   # Ornidex api key

# davinci:ft-personal-2023-08-22-12-58-39 : a fine-tuned model trained on an english dataset of 10 samples "data_prepared.jsonl"
# davinci:ft-personal-2023-08-22-15-13-42 : a fine-tuned model trained on a french dataset of 50 samples "data50_prepared.jsonl"
def chat_with_mydavinci(prompt):
    try:
         # Call the OpenAI API using the /v1/completions endpoint
        response = openai.Completion.create(
            engine="davinci:ft-personal-2023-08-22-12-58-39",  # You can change the model if needed
            prompt= prompt,
            max_tokens= 50
        )

        # Extract the assistant's reply from the response
        reply = response['choices'][0].text
        return reply
    except Exception as e:
        return str(e)


def chat_with_mygpt(prompt):
    try:
        # Call the OpenAI API using the /v1/chat/completions endpoint
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the assistant's reply from the response
        reply = completion.choices[0].message
        return reply
    except Exception as e:
        return str(e)


def main():
    print("Welcome to our fine-tuned BlaGPT!")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break

        # Call the chat_with_gpt function to get the assistant's reply
        assistant_reply = chat_with_mydavinci(user_input)

        print("BlaGPT:", assistant_reply)


if __name__ == "__main__":
    main()

