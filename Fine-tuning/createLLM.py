# _____________________ A LEGACY -- OLD OPENAI MODELS

# ON COMMAND LINE
# Linux/MacOS: set/export OPENAI_API_KEY = "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"
# windows: $env:OPENAI_API_KEY="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"

# data preparation: openai tools fine_tunes.prepare_data -f data.jsonl
# creating the FT model: openai api fine_tunes.create -t data_prepared.jsonl -m davinci


# _____________________ Innovation -- NEW OPENAI MODELS
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# STEP-1: uploading the training file
data = openai.File.create(
    file=open("..\Datasets\Raw\data_pcp.jsonl", "rb"),
    purpose='fine-tune'
)
print(data.id) # ex: file-upOvhQueR8ziH8CsGS8oFOia

# STEP-2: creating the model
openai.FineTuningJob.create(training_file="file-upOvhQueR8ziH8CsGS8oFOia", model="gpt-3.5-turbo")
# There's a bug issue on OpenAi server || bug reported --> waiting for a response


