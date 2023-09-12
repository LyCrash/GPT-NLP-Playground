from flask import Flask, request
from FineTuning.main import chat_with_mygpt_ctx_asc
#from Contextualisation.contextualisation_functions import contextualisation_function

app = Flask(__name__)

@app.route('/fine-tuning', methods=['POST'])
def fine_tuning_endpoint():
    data = request.json
    prompt = data.get('prompt')  # Access 'prompt' key
    context = data.get('context')  # Access 'context' key
    result = chat_with_mygpt_ctx_asc(prompt, context)
    return result

# @app.route('/contextualisation')
# def contextualisation_endpoint():
#     result = contextualisation_function()
#     return result

if __name__ == '__main__':
    app.run(debug=True)
