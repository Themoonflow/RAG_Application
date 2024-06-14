# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Assembling a RAG Application
# MAGIC
# MAGIC In this lab, we will assemble a Retrieval-augmented Generation (RAG) application using the components we previously created. The primary goal is to create a seamless pipeline where users can ask questions, and our system retrieves relevant documents from a Vector Search index to generate informative responses.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Setup the Retriever Component
# MAGIC
# MAGIC * **Task 2 :** Setup the Foundation Model
# MAGIC
# MAGIC * **Task 3 :** Assemble the Complete RAG Solution
# MAGIC
# MAGIC * **Task 4 :** Save the Model to Model Registry in Unity Catalog
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC
# MAGIC **🚨 Important:** This lab relies on the resources established in the previous one. Please ensure you have completed the prior lab before starting this one.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.16 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.9.0 databricks-sdk==0.12.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-Lab

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Setup the Retriever Component
# MAGIC **Steps:**
# MAGIC 1. Define the embedding model.
# MAGIC 1. Get the vector search index that was created in the previous lab.
# MAGIC 1. Generate a **retriever** from the vector store. The retriever should return **three results.**
# MAGIC 1. Write a test prompt and show the returned search results.
# MAGIC
# MAGIC
# MAGIC **🚨 Note: You need the Vector Search endpoint and index created before moving forward. These were created in the previous lab.**
# MAGIC

# COMMAND ----------

# Components we created before
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name = vs_endpoint_prefix+str(get_fixed_integer(DA.unique_name("_")))
print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

vs_index_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_self_managed_vs_index"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

# Define embedding model
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# test your retriever
vectorstore = get_retriever()
similar_documents = vectorstore.invoke("How Generative AI impacts humans?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Setup the Foundation Model
# MAGIC **Steps:**
# MAGIC 1. Define the foundation model for generating responses.
# MAGIC 2. Test the foundation model to ensure it provides accurate responses.

# COMMAND ----------

# Import necessary libraries
from langchain.chat_models import ChatDatabricks

# Define foundation model for generating responses
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 300)

# Test foundation model
print(f"Test chat model: {chat_model.predict('What is Generative AI?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3: Assemble the Complete RAG Solution
# MAGIC **Steps:**
# MAGIC 1. Merge the retriever and foundation model into a single Langchain chain.
# MAGIC 2. Configure the Langchain chain with proper templates and context for generating responses.
# MAGIC 3. Test the complete RAG solution with sample queries.

# COMMAND ----------

#  Import necessary libraries
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Define template for prompt
TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# Merge retriever and foundation model into Langchain chain
chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Test the complete RAG solution with sample query
question = {"query": "How Generative AI impacts humans?"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Save the Model to Model Registry in Unity Catalog
# MAGIC **Steps:**
# MAGIC 1. Register the assembled RAG model in the Model Registry with Unity Catalog.
# MAGIC 2. Ensure that all necessary dependencies and requirements are included.
# MAGIC 3. Provide an input example and infer the signature for the model.

# COMMAND ----------

# Import necessary libraries
from mlflow.models import infer_signature
import mlflow
import langchain

# Set Model Registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_demo4"

# Register the assembled RAG model in Model Registry with Unity Catalog
with mlflow.start_run(run_name="rag_app_demo4") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned how to assemble a Retrieval-augmented Generation (RAG) application using Databricks components. By integrating Vector Search for document retrieval and a foundational model for response generation, you created a powerful tool for answering user queries. This lab provided hands-on experience in building end-to-end AI applications and demonstrated the capabilities of Databricks for natural language processing tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>