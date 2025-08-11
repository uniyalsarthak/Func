- create a pipline along with rag which saves user and llm reponses and after a threshold it finetune the llm , automatically (half done) ✅ most imp
- improving the UI a lot lot and improve caching implemetation , spitting out proper syntaxed responses ,
- saving the context of page (searching query in page ) , while chatting (searching in the relevant page and also use it as a context ) - think about it

   <!-- requests from the RAG pipeline.
       * Advanced Retrieval Strategies:
           * Action: Implement re-ranking (e.g., using a cross-encoder model like Cohere Rerank or
             Sentence-BERT) to improve the relevance of retrieved chunks. Retrieve more than TOP_K initially,
             then re-rank to select the best TOP_K.
           * Action: Explore hybrid search (combining keyword search with vector search) for better recall,
             especially for very specific queries.
           * Action: Implement query expansion/rewriting (e.g., using an LLM to generate multiple versions of
             the user's query or to extract key entities) to improve retrieval effectiveness.
           * Action: Utilize metadata filtering during retrieval (e.g., "Show me information about scholarships
             for international students in 2025"). -->

        <!-- * Conversational RAG:
           * Action: Implement conversation history management. Pass previous turns of the conversation to the
             LLM to maintain context and enable follow-up questions.
           * Action: Use an LLM to summarize previous turns or extract key entities from the conversation to
             improve retrieval for subsequent queries.
       * Agentic RAG:
           * Action: For complex queries, consider using an LLM as an "agent" that can decide to use different
             tools (e.g., search the web, query a database, call a specific API) in addition to your RAG
             pipeline. -->
