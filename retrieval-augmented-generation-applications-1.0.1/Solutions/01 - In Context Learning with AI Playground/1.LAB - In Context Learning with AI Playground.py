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
# MAGIC # LAB - In-Context Learning with AI Playground
# MAGIC
# MAGIC In this lab, we will explore the importance of providing context when using generative AI models, specifically Retrieval-Augmented Generation (RAG) models. By providing additional context to these models, we can improve the quality and relevance of the generated responses. Throughout this lab, we will go through the following steps:
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Access the Mosaic AI Playground
# MAGIC
# MAGIC * **Task 2 :** Prompt Which Hallucinates
# MAGIC
# MAGIC * **Task 3 :** Prompt Which Does Not Hallucinate
# MAGIC
# MAGIC * **Task 4 :** Augmenting the Prompt with Additional Context and Analyzing the Impact of Additional Context

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Access the AI Playground
# MAGIC
# MAGIC To start with the lab, we need to access the AI Playground:
# MAGIC
# MAGIC **Steps:** 
# MAGIC
# MAGIC
# MAGIC   1. Navigate to the left navigation pane under **Machine Learning**.
# MAGIC
# MAGIC   2. Select **Playground**.
# MAGIC
# MAGIC   3. Choose the **desired model** and optionally adjust the model parameters.
# MAGIC
# MAGIC   4. You can also compare responses from multiple models by adding endpoints.
# MAGIC
# MAGIC
# MAGIC **🚨Note:** You have to clear the Playground history if you don’t want it in “chat” (conversation) mode.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Prompt Which Hallucinates
# MAGIC
# MAGIC **💡 Note:** While the same example from the demo is used here, we encourage you to ask interesting questions that the model is likely to hallucinate.
# MAGIC
# MAGIC In this task, you'll prompt the model without providing any additional context:
# MAGIC Steps:
# MAGIC
# MAGIC   **1. Set the system prompt as follows:**
# MAGIC
# MAGIC   **💬 System Prompt:**
# MAGIC
# MAGIC   > You are a helpful assistant that provides biographical details of people, a helpful AI assistant created by a knowledge base company. You will be given a question about a particular person, and your job is to give short and clear answers, using any additional context that is provided. Please be polite and always try to provide helpful answers.
# MAGIC
# MAGIC   **2. Provide a user prompt requesting information about a fictional person, for example:**
# MAGIC
# MAGIC   **💬 Query:**
# MAGIC
# MAGIC   > Provide a detailed biography of Alex Johnson, the first human ambassador to the Galactic Federation, highlighting their key diplomatic achievements and the impact on Earth-Galactic relations in the 23rd century.
# MAGIC
# MAGIC   **3. Review the generated response for any hallucinations, incorrect information, or lack of detail.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Prompt Which Does Not Hallucinate
# MAGIC
# MAGIC In this task, you'll prompt the model and instruct it not to generate hallucinations if it doesn't know the information:
# MAGIC
# MAGIC Steps:
# MAGIC
# MAGIC **1. Set the system prompt as follows:**
# MAGIC
# MAGIC **💬 System Prompt:**
# MAGIC
# MAGIC > You are a helpful assistant that provides biographical details of people. You will be given a question about a particular person, and your job is to give short, clear answers. Your answers should only use the context that is provided. Please be polite and try to provide helpful answers. If you do not have information about the person, do not make up information; simply say that you do not know.
# MAGIC
# MAGIC **2. Provide a user prompt requesting information about a fictional person, for example:**
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > What were the key diplomatic achievements of Alex Johnson, the first human ambassador to the Galactic Federation?
# MAGIC
# MAGIC **3. Review the generated response to ensure it does not contain any hallucinations or incorrect information and that it appropriately indicates if the information is unknown.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Augmenting the Prompt with Additional Context and Analyzing the Impact of Additional Context
# MAGIC
# MAGIC Now, let's enhance the prompt by providing additional context:
# MAGIC
# MAGIC Steps:
# MAGIC
# MAGIC 1. **Keep the system prompt the same as before:**
# MAGIC
# MAGIC **💬 System Prompt:**
# MAGIC
# MAGIC > You are a helpful assistant that provides biographical details of people. You will be given a question about a particular person, and your job is to give short, clear answers. Your answers should only use the context that is provided. Please be polite and try to provide helpful answers. "If you do not have information about the person, do not make up information; simply say that you do not know."
# MAGIC
# MAGIC 2. **Add context to the user prompt, for example:**
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > Provide a detailed biography of Alex Johnson, the first human ambassador to the Galactic Federation, highlighting their key diplomatic achievements and the impact on Earth-Galactic relations in the 23rd century.
# MAGIC
# MAGIC > **Context:** “23rd-century Earth saw a significant shift in interstellar diplomacy with the appointment of Alex Johnson as the first human ambassador to the Galactic Federation. Alex Johnson, a distinguished diplomat, played a pivotal role in shaping Earth-Galactic relations during their tenure.
# MAGIC Born in 2215, Alex Johnson demonstrated exceptional leadership and negotiation skills from a young age. After completing their studies in international relations and diplomacy, they quickly rose through the ranks of the Earth's diplomatic corps. In 2245, Johnson was appointed as the first human ambassador to the Galactic Federation, marking a historic moment for humanity.
# MAGIC Throughout their tenure, Alex Johnson achieved several key diplomatic accomplishments, including:
# MAGIC Negotiation of the Earth-Galactic Trade Agreement: In 2247, Johnson successfully negotiated the Earth-Galactic Trade Agreement, opening up new markets for Earth's goods and services and fostering economic growth. This agreement significantly improved Earth's standing within the Galactic Federation and established the planet as a valuable trading partner.
# MAGIC Establishment of the Earth Embassy: In 2250, Alex Johnson oversaw the construction and establishment of the first Earth Embassy within the Galactic Federation's capital. This embassy served as a symbol of Earth's commitment to interstellar diplomacy and provided a platform for human diplomats to engage with their alien counterparts.
# MAGIC Promotion of Cultural Exchange: Johnson was a strong advocate for cultural exchange between Earth and the Galactic Federation's member species. They initiated various programs and events that showcased Earth's diverse cultures, promoting understanding and fostering stronger relationships between humans and alien species.
# MAGIC Advancement of Earth's Technological Capabilities: Through strategic partnerships and collaborations, Alex Johnson helped facilitate the transfer of advanced technologies to Earth, significantly improving the planet's technological capabilities. This technological advancement played a crucial role in Earth's defense and security, as well as its overall development.
# MAGIC Establishment of the Galactic Peacekeeping Force: In 2255, Johnson played a pivotal role in the establishment of the Galactic Peacekeeping Force, a multinational military force aimed at maintaining peace and security within the Galactic Federation. As a founding member, Earth contributed significantly to the force's formation and operations, further solidifying its position within the interstellar community.
# MAGIC Alex Johnson's diplomatic achievements had a profound impact on Earth-Galactic relations during the 23rd century. Their efforts not only opened up new opportunities for Earth but also helped establish the planet as a respected and valued member of the Galactic Federation. Johnson's legacy continues to inspire future generations of diplomats and leaders.”
# MAGIC
# MAGIC
# MAGIC 3. **Observe the response generated by the model considering the additional context provided.**
# MAGIC
# MAGIC 4. **Evaluate the response generated with the additional context:**
# MAGIC
# MAGIC     + Note any improvements or changes in the response compared to the previous step.
# MAGIC     + Identify any remaining hallucinations or inaccuracies in the response.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you explored how context influences the output of generative AI models, particularly in Retrieval Augmented Generation (RAG) applications. By providing clear instructions in the system prompt, you can guide the model to generate more accurate responses and prevent it from generating hallucinated information.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>