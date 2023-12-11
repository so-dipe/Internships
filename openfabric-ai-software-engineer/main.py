import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pinecone
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2") 

pinecone.init(      
	api_key='4f3b24c7-8e52-421d-a942-b9a67b9a3ef9',      
	environment='gcp-starter'      
)      
pinecone_index = pinecone.Index('science')

def get_context(query_text, k=3):
    # Generate an embedding for the query_text using your GPT-2 model
    query_embedding = model.get_input_embeddings()(tokenizer.encode(query_text, return_tensors="pt")).mean(dim=1).detach().numpy().tolist()[0]
    
    # Retrieve similar vectors from Pinecone
    matches = pinecone_index.query(vector=query_embedding, top_k=k)['matches']
    matches = [match["id"] for match in matches]
    
    context = ""
    for id in matches:
        if int(id) < 8300:
            context += f"Context {id}: {pd.read_parquet('/data/0_to_25000.parquet').iloc[int(id)]['text']} \n"
    
    return context

############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    history = [{"role": "assistant", "message": "Hi! How can I help you?"}]  # Initialize history
    for text in request.text:
        system_prompt = (
            "System: I'm a science-savvy chatbot assistant here to tackle your scientific queries. "
            "I strive to provide accurate and concise answers, leveraging historical context to "
            "enhance our conversation."
        )
        context = get_context(text)

        combined_input = f"{system_prompt}\nHistory: {history}\n{context}\nUser: {text}"   
        
        input_ids = tokenizer.encode(combined_input, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        output.append(response)

        history.append({"role": "user", "message": text})
        history.append({"role": "assistant", "message": response})

    return SchemaUtil.create(SimpleText(), dict(text=output))
