import argparse
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"

# prompt template structure
PROMPT_TEMPLATE = '''
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
'''

# main function definition
def main():

    # parsing the commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text,")
    args = parser.parse_args()
    query_text = args.query_text

    # initializing the embeddings model and database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
    )

    # search the chroma db to find the similar results
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        # print(results)
        # print("\n\n")
        print("Unable to find matching results")
        return

    # preparing the prompt
    context_text = "\n\n---\n\n".join([ doc.page_content for doc, _score in results ])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question=query_text)
    print(prompt)

    #  initializing the model to predict the response
    model = ChatOllama(model="llama3.2")
    response_text = model.invoke(prompt)

    # preparing the result sources from document metadata
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"

    print(formatted_response)


# main starts here
if __name__ == "__main__":
    main()