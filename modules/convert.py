import csv
import jsonlines # -> takes care of formatting the strings, \n, ".."

# data preparation for OpenAi Fine-tuning

# DAVINCI: convert a csv database to a JSONL file (prompt-completion pairs = PCP)
def csv2jsonl_pcp(filename):
    with open(filename+".csv", "r", encoding="utf-8") as file:
        f_in = csv.reader(file)
        next(f_in) # skip the header row
        with jsonlines.open(filename+"_pcp.jsonl","w") as f_out:
            for (qst, rep) in f_in:
                entry = {
                    "prompt": qst,
                    "completion": rep
                }
                f_out.write(entry)


# GPT: convert a csv database to a JSONL file (the conversational chat format = CCF)
def csv2jsonl_ccf(filename, systemRole):
    with open(filename+".csv", "r", encoding="utf-8") as file:
        f_in = csv.reader(file)
        next(f_in) # skip the header row

        with jsonlines.open(filename+"_ccf.jsonl","w") as f_out:
            system = {
                    "role": "system",
                    "content": systemRole
                }
            for (qst, rep) in f_in:
                user = {
                    "role": "user",
                    "content": qst
                }
                assistant = {
                    "role": "assistant",
                    "content": rep

                }
                entry = {
                    "messages": [system, user, assistant]
                }
                f_out.write(entry)


#csv2jsonl_ccf ("data","You are a kind helpful BlaBlaCar customer-support assistant")