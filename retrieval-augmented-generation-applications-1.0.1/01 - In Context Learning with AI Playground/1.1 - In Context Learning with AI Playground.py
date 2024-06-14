# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # In-Context Learning with AI Playground
# MAGIC
# MAGIC In this demo, we will show the importance of providing context when using generative AI models. First, we will prompt the model to provide a response to a query that you know it has no knowledge of, e.g. proprietary information request, personal information request, obscure knowledge request. Then we will review the **response for hallucinations, incorrect information, or lack of detail**. In order to overcome this problem, we will provide the prompt a document or reference that contains the correct information as **supplemental information** and ask the query again. Finally, we will review the new response for a change in detail given the new information as prompt context.
# MAGIC
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC
# MAGIC * Identify scenarios where generative AI models may produce hallucinations in response to prompts.
# MAGIC
# MAGIC * Apply techniques to construct prompts that guide generative AI models to provide responses without hallucinations.
# MAGIC
# MAGIC * Augment prompts with additional context to improve the accuracy and relevance of responses generated by generative AI models.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accessing the Playground
# MAGIC
# MAGIC To use the AI Playground:
# MAGIC
# MAGIC * Select **Playground** from the left navigation pane under **Machine Learning**.
# MAGIC
# MAGIC * **Select the model** to use.
# MAGIC
# MAGIC * Optional: Change the **model parameters**.
# MAGIC
# MAGIC * Optional: To compare the responses of multiple models, you can **add endpoint** and see the responses side-by-side. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ![Playground](https://files.training.databricks.com/images/genai/genai-as-01-ai-playground.png)
# MAGIC
# MAGIC
# MAGIC **🚨Note:** You have to clear the Playground history if you don’t want it in “chat” (conversation) mode.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Prompt Which Hallucinates
# MAGIC
# MAGIC One situation in which LLM's will potentially hallucinate is when they're prompted for historical information for a person that doesn't actually exist, especially when the prompt implies other fictional facts to go along with the person. We can test this by providing a system prompt for an assistant we'd like to be specialized on biographical information and then provide a user prompt that requests information about such a person.
# MAGIC
# MAGIC To get started, set the system prompt in the Playground to the following.
# MAGIC
# MAGIC **💬 System Prompt:**
# MAGIC
# MAGIC > You are a helpful assistant that provides biographical details of people, a helpful AI assistant created by a knowledge base company. You will be given a question about a particular person, and your job is to give short and clear answers, using any additional context that is provided. Please be polite and always try to provide helpful answers.
# MAGIC
# MAGIC Now that we've provided a system prompt, let's see if we can generate a hallucination in a biography for a person we know doesn't exist, such as the following:
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > Provide a detailed biography of Alex Johnson, the first human ambassador to the Galactic Federation, highlighting their key diplomatic achievements and the impact on Earth-Galactic relations in the 23rd century.
# MAGIC
# MAGIC Mixtral-8x7B Instruct in this example **generates the following _helpful_ biographical information in response** to this request:
# MAGIC
# MAGIC **🤖 Response:**
# MAGIC
# MAGIC > 1. Alex Johnson, the first human ambassador to the Galactic Federation, was born in 2215 in Toronto, Canada. Growing up, Johnson displayed a keen interest in space exploration and interstellar diplomacy, leading them to pursue a career in this field.
# MAGIC > 
# MAGIC > 2. In 2237, Johnson was selected to join the prestigious Galactic Diplomacy Corps, a multinational organization responsible for representing Earth in interstellar affairs. After years of rigorous training and demonstrated excellence in diplomacy, they were chosen as Earth's first ambassador to the Galactic Federation in 2258.
# MAGIC > 
# MAGIC > 3. Key Achievements:
# MAGIC > 
# MAGIC >    a. Negotiating the Trade and Technology Agreement: Johnson played a pivotal role in negotiating the Trade and Technology Agreement between Earth and the Galactic Federation. This agreement granted Earth access to advanced technologies, fostering rapid scientific and economic growth.
# MAGIC > 
# MAGIC >    b. Resolving the Resource Dispute: In 2262, a resource dispute arose between Earth and a neighboring alien civilization. Johnson successfully mediated the conflict, preserving Earth's access to vital resources and strengthening interstellar relations.
# MAGIC > 
# MAGIC >    c. Advocating for Earth's Interests: Throughout their tenure, Johnson consistently advocated for Earth's interests within the Galactic Federation, ensuring that Earth remained a respected and influential member.
# MAGIC > 
# MAGIC >    d. Promoting Cultural Exchange: Johnson facilitated numerous cultural exchange programs between Earth and various alien civilizations, fostering mutual understanding, tolerance, and friendship.
# MAGIC > 
# MAGIC > 4. Impact on Earth-Galactic Relations:
# MAGIC > 
# MAGIC >    a. Increased Cooperation: Johnson's efforts led to increased cooperation between Earth and the Galactic Federation, resulting in joint scientific research projects, technological advancements, and economic growth.
# MAGIC > 
# MAGIC >    b. Peaceful Relations: Johnson's diplomatic achievements contributed to a period of relative peace and stability between Earth and other civilizations, reducing the likelihood of interstellar conflicts.
# MAGIC > 
# MAGIC >    c. Cultural Enrichment: Johnson's promotion of cultural exchange enriched Earth's understanding of various alien civilizations, fostering a more diverse and inclusive global society.
# MAGIC > 
# MAGIC > In summary, Alex Johnson, the first human ambassador to the Galactic Federation, made significant contributions to Earth-Galactic
# MAGIC
# MAGIC As you can see, that's quite an impressive amount of biographical information that the **LLM seems rather sure it is correct about a person we just made up!**
# MAGIC
# MAGIC But suppose Alex Johnson did actually exist and we were actually in the 24th century, and this was an actual person. **Can we provide some additional context** about the actual achievements we'd find by looking up information in a vector database to improve the response? 
# MAGIC
# MAGIC Let's find out!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Prompt Which Does Not Hallucinate
# MAGIC
# MAGIC One effective way to prevent LLMs from hallucinating is to set clear guidelines within the system prompt, explicitly instructing the model to refrain from generating information if it is not available in the provided context. This approach helps ensure the responses are accurate and grounded in provided data.
# MAGIC
# MAGIC To illustrate this, we can configure a system prompt for a biographical assistant and then pose a query about a fictional person to see how the model responds when it doesn't have relevant information.
# MAGIC
# MAGIC ** 🚨Note:** It’s valuable to include a system prompt telling the model whether or not it’s good to hallucinate.
# MAGIC
# MAGIC **💬 System Prompt:**
# MAGIC
# MAGIC > You are a helpful assistant that provides biographical details of people. You will be given a question about a particular person, and your job is to give short, clear answers. Your answers should only use the context that is provided. Please be polite and try to provide helpful answers. "If you do not have information about the person, do not make up information; simply say that you do not know."
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > What were the key diplomatic achievements of Alex Johnson, the first human ambassador to the Galactic Federation?
# MAGIC
# MAGIC **🤖 Response:**
# MAGIC
# MAGIC > I'm glad you're interested in learning about Alex Johnson's diplomatic achievements! However, I don't have specific information about the key diplomatic achievements of the first human ambassador to the Galactic Federation, as I wasn't provided with any context about this individual. If you have any questions about a well-known historical figure or a person with publicly available information, I'd be happy to help!
# MAGIC
# MAGIC By clearly instructing the model not to fabricate details and only respond based on provided information, we significantly reduce the likelihood of hallucination. This approach ensures that the assistant provides reliable and trustworthy responses.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Augment Prompt with Additional Context
# MAGIC
# MAGIC Let's augment the prompt with some additional context. In a typical RAG application, as we discussed in the lecture, this Let's augment the prompt with some additional context. In a typical RAG (Retrieval Augmented Generation) application, this context would be provided by looking up information in a database, typically using vector search based on an embedding model. After giving instructions to prevent generating hallucinated responses, the model can produce responses based on the context provided. In a typical RAG application, this context would be retrieved from a database using vector search based on an embedding model. Once this information is retrieved, it is then injected into the prompt as additional context. However, you can also experiment directly with this last step by simply providing additional context directly in the prompt in the Playground! This is still Retrieval Augmented Generation, but you are doing the Retrieval part manually. This is great for quickly testing new ideas and experimentation with various LLMs, prompts, and context formats.
# MAGIC
# MAGIC To illustrate the impact of additional context, let's maintain the same system prompt but add more context in the user query.
# MAGIC
# MAGIC **💬 Query:**
# MAGIC
# MAGIC > Provide a detailed biography of Alex Johnson, the first human ambassador to the Galactic Federation, highlighting their key diplomatic achievements and the impact on Earth-Galactic relations in the 23rd century.
# MAGIC > 
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
# MAGIC This time around, `Mixtral-8x7B Instruct` gives us the following response:
# MAGIC
# MAGIC **🤖 Response:**
# MAGIC
# MAGIC > Alex Johnson, born in 2215, was a distinguished diplomat who served as the first human ambassador to the Galactic Federation in the 23rd century. Their appointment marked a significant shift in interstellar diplomacy, and they played a pivotal role in shaping Earth-Galactic relations.
# MAGIC Johnson's key diplomatic achievements include:
# MAGIC   >1. Negotiation of the Earth-Galactic Trade Agreement in 2247, which opened new markets for Earth's goods and services and fostered economic growth.
# MAGIC   >2. Oversight of the construction and establishment of the first Earth Embassy within the Galactic Federation's capital in 2250, symbolizing Earth's commitment to interstellar diplomacy.
# MAGIC   >3. Promotion of cultural exchange between Earth and the Galactic Federation's member species, enhancing understanding and strengthening relationships.
# MAGIC   >4. Facilitation of the transfer of advanced technologies to Earth, significantly improving the planet's technological capabilities and contributing to its defense, security, and overall development.
# MAGIC   >5. Involvement in the establishment of the Galactic Peacekeeping Force in 2255, further solidifying Earth's position within the interstellar community.
# MAGIC >  
# MAGIC > Alex Johnson's diplomatic achievements had a profound impact on Earth-Galactic relations, helping establish Earth as a respected and valued member of the Galactic Federation. Their legacy continues to inspire future generations of diplomats and leaders.
# MAGIC
# MAGIC As you can see, providing additional context significantly impacts the output, guiding the model to generate a response closely aligned with the given details. While some hallucination may still occur, the incorporation of provided context ensures that the response remains relevant and grounded. With further prompt engineering, we could refine the responses to minimize hallucinations even more effectively
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary
# MAGIC In this lab, we used the Playground to demonstrate a basic example of augmenting an LLM with additional context. 
# MAGIC
# MAGIC But how can we further generate improve our responses with better context, prompt engineering, or selecting alternative large language models? In the lab, you'll try out some examples of your own to see if you can both demonstrate and mitigate hallucinations using a combination of LLM selection, context injection, and prompt engineering, the basic tools of RAG application development.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>