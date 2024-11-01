__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key = key)
g_embed = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

#genai.configure(api_key = key) 


filenames = ["AlarmClock", "Headphones", "IceBucket", "WashingMachine"]
options = ["Sangean Clock Radio", "Audio-Technica Headphones", "Frigidaire Ice Machine", "WonderWash Washing Machine"]


productNames = ["Sangean Digital Clock Radio", "Audio-Technica Headphones", "Frigidaire Ice Machine", "WonderWash Portable Washing Machine"]

productLinks = ["https://www.amazon.com/Sangean-RCR-5-Digital-Clock-Radio/dp/B0016CWV3U?th=1", "https://www.amazon.com/Audio-Technica-ATH-M30-Closed-Back-Headphones/dp/B00007E7C8", "https://www.amazon.com/Frigidaire-EFIC103-Machine-Icemaker-Stainless/dp/B004VV8GOQ",  "https://www.amazon.com/WonderWash-Portable-Washing-Machine-Apartment/dp/B002C8HR9A"]



docs=[]
descriptions = []
vectordblist = []
retrievers = []

sentiment_vectordblist = []
sentiment_retrievers = []

i = 0
for file in filenames:

    #print(docs[100].page_content)
    dfile = open("./ProductDescriptions/"+file+".txt", "r")
    descriptions.append(dfile.read())

    #vectorstore = Chroma.from_documents(collection_name = file, documents=docs[i], embedding=g_embed, persist_directory="./rag_vectorstore") #use to create new vector db or append to existing db
    vectordblist.append(Chroma(collection_name = file, persist_directory="./rag_vectorstore", embedding_function=g_embed)) #use to access preexisting vector db
    
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50}) #use with new db to append
    retrievers.append( vectordblist[i].as_retriever(search_type="similarity", search_kwargs={"k": 50}) )
    
    i+=1

i=0
for file in filenames:

    #vectorstore = Chroma.from_documents(collection_name = file, documents=docs[i], embedding=g_embed, persist_directory="./rag_vectorstore") #use to create new vector db or append to existing db
    sentiment_vectordblist.append(Chroma(collection_name = file+"_Sentiment", persist_directory="./rag_vectorstore", embedding_function=g_embed)) #use to access preexisting vector db
    
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50}) #use with new db to append
    sentiment_retrievers.append( sentiment_vectordblist[i].as_retriever(search_type="similarity", search_kwargs={"k": 50}) )
    
    i+=1

#model = genai.GenerativeModel("gemini-1.5-flash")


k = 0
product = st.sidebar.radio("Product", options)

sentimentBool = True

if product == "Sangean Clock Radio":
    k = 0
if product == "Audio-Technica Headphones":
    k = 1
if product == "Frigidaire Ice Machine":
    k = 2
if product == "WonderWash Washing Machine":
    k = 3


st.header("ReviewerLLM: AI-Powered Feedback Summarizer", divider="gray")

st.subheader(productNames[k])

st.link_button("Link to Amazon Page", productLinks[k], icon=":material/link:")

with st.form("form"):
   user_input = st.text_area(
      "Enter prompt:",
   )
   submitted = st.form_submit_button("Enter")



if st.button("General Product Summary"):
    if(sentimentBool):
        system_prompt = (
            "You are a product analyst. Summarize the following reviews by identifying key strengths, weaknesses, "
            "and areas for improvement. Highlight what customers liked most and where there were consistent complaints "
            "or suggestions for enhancement. Make sure to include any specific product features or experiences that "
            "were praised or criticized. Each review starts with a sentiment analysis statement that tells you if the review is very negative, negative, neutral, positive, or very positive. Positive responses show that the user praised and liked aspects of the product, while negative responses show the user criticized and disliked aspects of the product. These sentiments can differentiate between which aspects were praised and which were criticized. Think about these sentiments and use it to improve your summary.\n\n"
            "Here is the product description for this product: "
            + descriptions[k] +
            "\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
        rag_chain = create_retrieval_chain(sentiment_retrievers[k], question_answer_chain)

        response = rag_chain.invoke()
        st.write(response["answer"])
        
    else:
        system_prompt = (
            "You are a product analyst. Summarize the following reviews by identifying key strengths, weaknesses, "
            "and areas for improvement. Highlight what customers liked most and where there were consistent complaints "
            "or suggestions for enhancement. Make sure to include any specific product features or experiences that "
            "were praised or criticized.\n\n"
            "Here is the product description for this product: "
            + descriptions[k] +
            "\n\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
        rag_chain = create_retrieval_chain(sentiment_retrievers[k], question_answer_chain)

        response = rag_chain.invoke()
        st.write(response["answer"])




if(submitted):
    if(sentimentBool):
        system_prompt = (
            "You are a product analyst."
            "Based on the information in the following product reviews, create a useful "
            "answer to the question about the product. Each review starts with a sentiment analysis statement that tells you if the review is very negative, negative, neutral, positive, or very positive. Positive responses show that the user praised and liked aspects of the product, while negative responses show the user criticized and disliked aspects of the product. These sentiments can differentiate between which aspects were praised and which were criticized. Think about these sentiments and use it to improve your answers. Consider different perspecives and different possibilities." 
            "Be detailed and concise in your answers. Write your response directly about the product, not about reviewers."
            "Here is the product description for this product:" 
            + descriptions[k] +
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
        rag_chain = create_retrieval_chain(sentiment_retrievers[k], question_answer_chain)

        response = rag_chain.invoke({'input': user_input})
        st.write(response["answer"])
        
    else:
        system_prompt = (
            "You are a product analyst."
            "Based on the information in the following product reviews, create a useful "
            "answer to the question about the product. Consider different perspecives and different possibilities." 
            "Be detailed and concise in your answers. Write your response objectively and directly about the product."
            "Here is the product description for this product:" 
            + descriptions[k] +
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
        rag_chain = create_retrieval_chain(retrievers[k], question_answer_chain)

        response = rag_chain.invoke({'input': user_input})
        st.write(response["answer"])

