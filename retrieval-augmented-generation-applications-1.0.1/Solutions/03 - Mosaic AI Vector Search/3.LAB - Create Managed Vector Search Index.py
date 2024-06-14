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
# MAGIC # LAB - Create a "Managed" Vector Search Index
# MAGIC
# MAGIC The objective of this lab is to demonstrate the process of creating a **managed** Vector Search index for retrieval-augmented generation (RAG) applications. This involves configuring Databricks Vector Search to ingest data from a Delta table containing text embeddings and metadata.
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Create a Vector Search endpoint to serve the index.
# MAGIC
# MAGIC * **Task 2 :** Connect Delta Table with Vector Search Endpoint
# MAGIC
# MAGIC * **Task 3 :** Test the Vector Search Index
# MAGIC
# MAGIC * **Task 4 :** Re-rank search results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC
# MAGIC **🚨 Important: This lab relies on the resources established in the previous Lab. Please ensure you have completed the prior lab before starting this lab.**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install -U transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0 flashrank==0.2.0
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
# MAGIC ## Task 1: Create a Vector Search Endpoint
# MAGIC
# MAGIC To start, you need to create a Vector Search endpoint to serve the index.
# MAGIC
# MAGIC **🚨IMPORTANT: Vector Search endpoints must be created before running the rest of the demo. Endpoint names should be in this format; `vs_endpoint_x`. The endpoint will be assigned by username.**
# MAGIC
# MAGIC **💡 Instructions:**
# MAGIC
# MAGIC 1. Define the endpoint that you will use if you don't have endpoint creation permissions. 
# MAGIC 1. [Optional]: Create a new enpoint. Check if the vector serch endpoint exists, if not, create it.
# MAGIC 1. Wait for the endpoint to be ready.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step-by-Step Instructions:
# MAGIC
# MAGIC 1. **Define Endpoint Name**: Set the variable `VECTOR_SEARCH_ENDPOINT_NAME` to the name of the endpoint you will use. If you don't have endpoint creation permissions, use the name of the existing endpoint.
# MAGIC
# MAGIC > - Additionally, you can check the endpoint status in the Databricks workspace under `Compute > Vector Search`.

# COMMAND ----------

# assign vs search endpoint by username
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name = vs_endpoint_prefix+str(get_fixed_integer(DA.unique_name("_")))
print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

vsc = VectorSearchClient()

# COMMAND ----------

# MAGIC %md
# MAGIC **(Optional): Create New Endpoint**:
# MAGIC     - If you have endpoint creation permissions:
# MAGIC         - Uncomment the provided code block to create the endpoint.
# MAGIC         - Run the code block to create the endpoint if it doesn't exist.
# MAGIC     - If you don't have permissions:
# MAGIC         - Skip this step.

# COMMAND ----------

# Verify if the vector search endpoint already exists, if not, create it
# try:
#     # Check if the vector search endpoint exists
#     vsc.get_endpoint(name=vs_endpoint_name)
#     print("Endpoint found: " + vs_endpoint_name)
# except Exception as e:
#     print("\nEndpoint not found: " + vs_endpoint_name)
#     # Create a new vector search endpoint
#     if "NOT_FOUND" in str(e):
#         print("\nCreating Endpoint...")
#         vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")
#     print("Endpoint Created: " + vs_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. **Wait for Endpoint to be Ready**:
# MAGIC     - After defining the endpoint name:
# MAGIC         - Check the status of the endpoint using the provided function `wait_for_vs_endpoint_to_be_ready`.
# MAGIC         >**Note:** 
# MAGIC         > - If you encounter an error indicating that the endpoint is not found, proceed with the following steps:
# MAGIC         >     - Replace the endpoint name in the code with the appropriate name.
# MAGIC         >     - Uncomment and run the code block provided to create the endpoint (if you have permissions).

# COMMAND ----------

# check the status of the endpoint
wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Create a Managed Vector Search Index
# MAGIC
# MAGIC Now, connect the Delta table containing text and metadata with the Vector Search endpoint. In this lab, you will create a **managed** index, which means you don't need to create the embeddings manually. For API details, check the [documentation page](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-index-using-the-python-sdk).
# MAGIC
# MAGIC **📌 NOTE: You will use the embeddings table that you created in the previous lab. If you haven't completed that lab, stop here and complete it first.**
# MAGIC
# MAGIC **💡 Instructions:**
# MAGIC
# MAGIC 1. Define the source Delta table containing the text to be indexed.
# MAGIC
# MAGIC 1. Create a Vector Search index. Use these parameters; Source column as `Title` and `databricks-bge-large-en` as embedding model. Also, the sync process should be  `manually` triggered.
# MAGIC
# MAGIC 1. Create or synchronize the Vector Search index based on the source Delta table.
# MAGIC

# COMMAND ----------

# The Delta table containing the text embeddings and metadata
source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_embeddings"

# The Delta table to store the Vector Search index
vs_index_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_self_managed_vs_index"

# create or sync the index
if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
  
  vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Search Documents Similar to the Query
# MAGIC
# MAGIC Test the Vector Search index by searching for similar content based on a sample query.
# MAGIC
# MAGIC **💡 Instructions:**
# MAGIC
# MAGIC 1. Get the index instance that we created.
# MAGIC
# MAGIC 1. Send a sample query to the language model endpoint using **query text**. 🚨 Note: As you created a managed index, you will use plain text for similarity search using `query_text` parameter.
# MAGIC
# MAGIC 1. Use the embeddings to search for similar content in the Vector Search index.

# COMMAND ----------

# get vs index
index = vsc.get_index(vs_endpoint_name, vs_index_fullname)

question = "What are the security and privacy concerns when training generative models?"

# search 
results = index.similarity_search(
    query_text = question,
    columns=["pdf_name", "content"],
    num_results=4
    )

# show the results
docs = results.get('result', {}).get('data_array', [])

pprint(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Re-rank Search Results
# MAGIC
# MAGIC You have retreived some documents that are similar to the query text. However, the question of which documents are the most relevant is not done by the vector search results. Use `flashrank` library to re-rank the results and show the most relevant top 3 documents. 
# MAGIC
# MAGIC **💡 Instructions:**
# MAGIC
# MAGIC 1. Define `flashrank` with **`rank-T5-flan`** model.
# MAGIC
# MAGIC 1. Re-rank the search results.
# MAGIC
# MAGIC 1. Show the most relevant **top 3** documents.
# MAGIC

# COMMAND ----------

from flashrank import Ranker, RerankRequest

# define the ranker
cache_dir=f"{DA.paths.working_dir.replace('dbfs:/', '/dbfs/')}/opt"
ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)

# format result to align with reranker lib format 
passages = []
for doc in docs:
    new_doc = {"file": doc[0], "text": doc[1]}
    passages.append(new_doc)

# rerank the passages
rerankrequest = RerankRequest(query=question, passages=passages)
ranked_passages = ranker.rerank(rerankrequest)

# show to 3 results
print (*ranked_passages[:3], sep="\n\n")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC **🚨 Warning:** Please don't delete the catalog and tables created in this lab as next labs depend on these resources. To clean-up the classroom assets, run the classroom clean-up script in the last lab.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned how to set up a Vector Search index using Databricks Vector Search for retrieval-augmented generation (RAG) applications. By following the tasks, you successfully created a Vector Search endpoint, connected a Delta table containing text embeddings, and tested the search functionality. Furthermore, using a re-ranking library, you re-ordered the search results from the most relevant to least relevant documents. This lab provided hands-on experience in configuring and utilizing Vector Search, empowering you to enhance content retrieval and recommendation systems in your projects.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>