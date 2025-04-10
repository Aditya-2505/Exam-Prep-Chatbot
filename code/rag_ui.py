from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from threading import Thread
import gradio as gr

stop_tokens = llm_model_configuration.get("stop_tokens")
rag_prompt_template = llm_model_configuration["rag_prompt_template"]


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = llm.pipeline.tokenizer.convert_tokens_to_ids(stop_tokens)

    stop_tokens = [StopOnTokens(stop_tokens)]


def load_single_document(file_path: str) -> List[Document]:
    """
    helper for loading a single document

    Params:
      file_path: document path
    Returns:
      documents loaded

    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")


def default_partial_text_processor(partial_text: str, new_text: str):
    """
    helper for updating partially generated answer, used by default

    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string

    """
    partial_text += new_text
    return partial_text


text_processor = llm_model_configuration.get("partial_text_processor", default_partial_text_processor)


def create_vectordb(
    docs, spliter_name, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, run_rerank, search_method, score_threshold, progress=gr.Progress()
):
    """
    Initialize a vector database

    Params:
      doc: orignal documents provided by user
      spliter_name: spliter method
      chunk_size:  size of a single sentence chunk
      chunk_overlap: overlap size between 2 chunks
      vector_search_top_k: Vector search top k
      vector_rerank_top_n: Search rerank top n
      run_rerank: whether run reranker
      search_method: top k search method
      score_threshold: score threshold when selecting 'similarity_score_threshold' method

    """
    global db
    global retriever
    global combine_docs_chain
    global rag_chain

    if vector_rerank_top_n > vector_search_top_k:
        gr.Warning("Search top k must >= Rerank top n")

    documents = []
    for doc in docs:
        if type(doc) is not str:
            doc = doc.name
        documents.extend(load_single_document(doc))

    text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embedding)
    if search_method == "similarity_score_threshold":
        search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
    else:
        search_kwargs = {"k": vector_search_top_k}
    retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
    if run_rerank:
        reranker.top_n = vector_rerank_top_n
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    prompt = PromptTemplate.from_template(rag_prompt_template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return "Vector database is Ready"


def update_retriever(vector_search_top_k, vector_rerank_top_n, run_rerank, search_method, score_threshold):
    """
    Update retriever

    Params:
      vector_search_top_k: Vector search top k
      vector_rerank_top_n: Search rerank top n
      run_rerank: whether run reranker
      search_method: top k search method
      score_threshold: score threshold when selecting 'similarity_score_threshold' method

    """
    global db
    global retriever
    global combine_docs_chain
    global rag_chain

    if vector_rerank_top_n > vector_search_top_k:
        gr.Warning("Search top k must >= Rerank top n")

    if search_method == "similarity_score_threshold":
        search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
    else:
        search_kwargs = {"k": vector_search_top_k}
    retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
    if run_rerank:
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
        reranker.top_n = vector_rerank_top_n
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return "Vector database is Ready"


def bot(history, temperature, top_p, top_k, repetition_penalty, hide_full_prompt, do_rag):
    """
    callback function for running chatbot on submit button click

    Params:
      history: conversation history
      temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
      top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      hide_full_prompt: whether to show searching results in promopt.
      do_rag: whether do RAG when generating texts.

    """
    streamer = TextIteratorStreamer(
        llm.pipeline.tokenizer,
        timeout=3600.0,
        skip_prompt=hide_full_prompt,
        skip_special_tokens=True,
    )
    pipeline_kwargs = dict(
        max_new_tokens=512,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    if stop_tokens is not None:
        pipeline_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

    llm.pipeline_kwargs = pipeline_kwargs
    if do_rag:
        t1 = Thread(target=rag_chain.invoke, args=({"input": history[-1][0]},))
    else:
        input_text = rag_prompt_template.format(input=history[-1][0], context="")
        t1 = Thread(target=llm.invoke, args=(input_text,))
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1][1] = partial_text
        yield history


def request_cancel():
    llm.pipeline.model.request.cancel()


# initialize the vector store with example document
create_vectordb(
    [text_example_path],
    "RecursiveCharacter",
    chunk_size=400,
    chunk_overlap=50,
    vector_search_top_k=10,
    vector_rerank_top_n=2,
    run_rerank=True,
    search_method="similarity_score_threshold",
    score_threshold=0.5,
)

if not Path("gradio_helper.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/llm-rag-langchain/gradio_helper.py")
    open("gradio_helper.py", "w").write(r.text)

from gradio_helper import make_demo

demo = make_demo(
    load_doc_fn=create_vectordb,
    run_fn=bot,
    stop_fn=request_cancel,
    update_retriever_fn=update_retriever,
    model_name=llm_model_id.value,
    language=model_language.value,
)

try:
    demo.queue().launch()
except Exception:
    demo.queue().launch(share=True)
# If you are launching remotely, specify server_name and server_port
# EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
# To learn more please refer to the Gradio docs: https://gradio.app/docs/