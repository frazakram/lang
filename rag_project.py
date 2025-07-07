from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
load_dotenv()
embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=300)
documents=[

    "Virat kohli is the indian batsman",
    "Sachin Tendulkar is the indian batsman",
    "Ricky Ponting is the Australian batsman",
]
query="who is Virat kohli ?"
result=embedding.embed_query(query)
result2=embedding.embed_documents(documents)
# print(result2)
# print(result)
scores=cosine_similarity([result],result2)[0]
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print(f"similiarity score is {score}")
