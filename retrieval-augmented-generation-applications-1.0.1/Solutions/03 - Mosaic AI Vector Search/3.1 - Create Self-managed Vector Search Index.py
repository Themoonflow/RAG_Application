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
# MAGIC # Create a "Self-Managed" Vector Search Index
# MAGIC
# MAGIC In the previous demo, we chunked the raw PDF document pages into small sections, computed the embeddings, and saved it as a Delta Lake table. Our dataset is now ready. 
# MAGIC
# MAGIC Next, we'll configure Databricks Vector Search to ingest data from this table.
# MAGIC
# MAGIC Vector search index uses a Vector search endpoint to serve the embeddings (you can think about it as your Vector Search API endpoint). <br/>
# MAGIC Multiple Indexes can use the same endpoint. Let's start by creating one.
# MAGIC
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Insect the Vector Search endpoint and index using the UI. 
# MAGIC
# MAGIC * Set up an endpoint for Vector Search.
# MAGIC
# MAGIC * Store the embeddings and their metadata using the Vector Search.
# MAGIC
# MAGIC * Retrieve documents from the vector store using similarity search.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC
# MAGIC
# MAGIC **🚨 Important:** This demonstration relies on the resources established in the previous one. Please ensure you have completed the prior demonstration before starting this one.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U --quiet transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.16 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.9.0 flashrank==0.2.0
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-03

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
# MAGIC ## Demo Overview
# MAGIC
# MAGIC As seen in the diagram below, in this demo we will focus on the Vector Search indexing section (highlighted in orange).  
# MAGIC
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/genai/genai-as-01-rag-pdf-self-managed-3.png" width="100%">
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a "Self-Managed" Vector Search Index
# MAGIC
# MAGIC Setting up a Databricks Vector Search index involves a few key steps. First, you need to decide on the method of providing vector embeddings. Databricks supports three options: 
# MAGIC
# MAGIC * providing a source Delta table containing text data
# MAGIC * **providing a source Delta table that contains pre-calculated embeddings**
# MAGIC * using the Direct Vector API to create an index on embeddings stored in a Delta table
# MAGIC
# MAGIC In this demo, we will go with the first method. 
# MAGIC
# MAGIC Next, we will **create a vector search endpoint**. And in the final step, we will **create a vector search index** from a Delta table. 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup a Vector Search Endpoint
# MAGIC
# MAGIC The first step for creating a Vector Search index is to create a compute endpoint. This endpoint serves the vector search index. You can query and update the endpoint using the REST API or the SDK. 
# MAGIC
# MAGIC **🚨IMPORTANT: Vector Search endpoints must be created before running the rest of the demo. Endpoint names should be in this format; `vs_endpoint_x`. The endpoint will be assigned by username.**
# MAGIC
# MAGIC **💡 Note: In the code block below, `create_endpoint` function is commented out. If you have right permissions, you can create an endpoint using that function.**

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


vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# IF YOU HAVE ENDPOINT CREATION PERMISSIONS, UNCOMMENT THIS CODE AND RUN IT TO CREATE AN ENDPOINT
# if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
#     vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

# COMMAND ----------

# check the status of the endpoint
wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the Endpoint
# MAGIC
# MAGIC After the endpoint is created, you can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect Delta Table with Vector Search Endpoint
# MAGIC
# MAGIC After creating the endpoint, we can create the **vector search index**. The vector search index is created from a Delta table and is optimized to provide real-time approximate nearest neighbor searches. The goal of the search is to identify documents that are similar to the query. 
# MAGIC
# MAGIC **Vector search indexes appear in and are governed by Unity Catalog.**

# COMMAND ----------

#The table we'd like to index
source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_embeddings"

# Where we want to store our index
vs_index_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_self_managed_vs_index"

# create or sync the index
if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for Similar Content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC **📌 Note:** `similarity_search` also supports a filter parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).
# MAGIC

# COMMAND ----------

import mlflow.deployments


deploy_client = mlflow.deployments.get_deploy_client("databricks")
question = "How Generative AI impacts humans?"
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

# get similar 5 documents
results = vsc.get_index(vs_endpoint_name, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["pdf_name", "content"],
  num_results=5)

# format result to align with reranker lib format 
passages = []
for doc in results.get('result', {}).get('data_array', []):
    new_doc = {"file": doc[0], "text": doc[1]}
    passages.append(new_doc)

pprint(passages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-ranking Search Results
# MAGIC
# MAGIC For re-ranking the results, we will use a very light library. [**`flashrank`**](https://github.com/PrithivirajDamodaran/FlashRank) is an open-source reranking library based on SoTA cross-encoders. The library supports multiple models, and in this example we will use `rank-T5` model. 
# MAGIC
# MAGIC After re-ranking you can review the results to check if the order of the results has changed. 
# MAGIC
# MAGIC **💡Note:** Re-ranking order varies based on the model used!

# COMMAND ----------

from flashrank import Ranker, RerankRequest


ranker = Ranker(model_name="rank-T5-flan", cache_dir=f"{DA.paths.working_dir.replace('dbfs:/', '/dbfs/')}/opt")

rerankrequest = RerankRequest(query=question, passages=passages)
results = ranker.rerank(rerankrequest)
print (*results[:3], sep="\n\n")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC **🚨 Warning:** Please refrain from deleting the catalog and tables created in this demo, as they are required for upcoming demonstrations. To clean up the classroom assets, execute the classroom clean-up script provided in the final demo.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, the objective was to generate embeddings from documents and store them in Vector Search. The initial step involved creating a Vector Search index, which required the establishment of a compute endpoint and the creation of an index that is synchronized with a source Delta table. Following this, we conducted a search for the stored indexes using a sample input query.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>