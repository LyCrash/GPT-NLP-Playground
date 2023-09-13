from flask import Flask, request
from FineTuning.main import chat_with_mygpt_ctx_asc
from Contextualisation.main import chat_with_gpt_embed

app = Flask(__name__)

@app.route('/fine-tuning', methods=['POST'])
def fine_tuning_endpoint():
    # get the data parameters from the request body
    data = request.json

    # get the user input
    prompt = data.get('prompt') 
    # get the context of the discussion (depends on previous prompts)
    context = data.get('context') 

    # calling the fine-tuned gpt-3.5-turbo model that takes into account the chat context (approche de fine-tuning avec contextualisation ascendante )
    result = chat_with_mygpt_ctx_asc(prompt, context)
    return result

@app.route('/contextualisation', methods=['POST'])
def contextualisation_endpoint():
    # Give the role to the chatbot
    title = "Tu es un expert en assistance clientèle bienveillant chez BlaBlaCar, la plateforme de covoiturage en ligne"
    # The file used as a source
    csv_filename = "Datasets/Raw/data.csv"

    # get the data parameters from the request body
    data = request.json

    # Number of question-answer pairs wanted
    top_n = data.get('top_n')
    # get the user input
    prompt = data.get('prompt') 
    # get the context of the discussion (depends on previous prompts)
    context = data.get('context') 

    # call the generic gpt-3.5-turbo model with a personnalised context using sentence embeddings (method3)
    result = chat_with_gpt_embed(title, prompt, context, csv_filename, top_n)
    return result

if __name__ == '__main__':
    app.run(debug=True)
