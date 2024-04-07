import argparse
import atexit
from typing import List

import box
import sys
import yaml

import logging.config
import logging.handlers

from pathlib import Path
from datetime import datetime
from pypdf import PdfReader

from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import MultiModalRetriever, PreProcessor, EmbeddingRetriever

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama


logger = logging.getLogger(__name__)

with open('configs/config.yml', 'r', encoding='utf8') as ymlfile:
    config = box.Box(yaml.safe_load(ymlfile))


def setup_logging():
    config_file = Path("configs/log_config.yml")
    with open(config_file) as log_file:
        log_config = yaml.safe_load(log_file)
    logging.config.dictConfig(log_config)
    queue_handler = logging.getHandlerByName("queue_handler")
    if queue_handler is not None:
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


def process_pdf(pdf_path: str) -> str:
    """Function which takes a path to a pdf and process it.
    Returns a list of Document objects, each containing a page out of the input pdf.

    :param pdf_path: path to the PDF.
    :returns: A list of Document objects
    """
    pdf_path = Path.cwd() / pdf_path

    if not Path(pdf_path).exists():
        logger.error(f"File not found")
        return

    if not Path(pdf_path).suffix == ".pdf":
        logger.error(f"File not a pdf")
        return

    pdf_reader = PdfReader(pdf_path)

    docs_list = []
    for idx, page in enumerate(pdf_reader.pages):
        docs_list.append(Document(content=f"{page.extract_text()}",
                              meta={"filename": pdf_path.name, "page": idx + 1, "date_added": datetime.now().isoformat()}))

    return docs_list


def page_chunker(doc: Document) -> List[Document]:
    """Function which chunks up a pdf page into shortened Document objects used for retrieval.

    :param doc: A Document representing the content of a page out of the pdf.
    :returns: a list of Document objects
    """

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
        progress_bar=False
    )
    get_page_number = doc.meta["page"]

    # Process the documents with the PreProcessor
    preprocessed_docs = preprocessor.process([doc])
    logger.debug(f"Completed preprocessing/chunking document with page number: {get_page_number}.")
    logger.debug(f"Number of chunked documents is: {len(preprocessed_docs)}.")
    return preprocessed_docs


def get_retriever(document_store):
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=config.retriever.embedding_model,
        model_format=config.retriever.model_format,
        use_gpu=config.retriever.use_gpu,
        progress_bar=False,
        top_k=config.retriever.top_k,
        scale_score=False
    )
    return retriever


# def chatter(prompt):
#     model = Path.home() / "models" / config.llm.model
#     llm = Llama(
#         model_path=str(model),
#         n_ctx=config.llm.context,
#         n_threads=config.llm.threads,
#         n_gpu_layers=config.llm.gpu_layers,
#         temperature=config.llm.temperature,
#         stream=config.llm.stream
#     )
#
#     prompt_input = f"""
#     {prompt}
#     """
#
#     prompt_setup = f"""
#     <|im_start|>system
#     You are an assistant. You are here to help<|im_end|>
#     <|im_start|>user
#     {prompt_input}<|im_end|>
#     <|im_start|>assistant
#     """
#     llm.reset()
#     result = llm(prompt=prompt_setup, max_tokens=128, stream=True)
#
#     for chunk in result:
#         print(chunk['choices'][0]['text'], end='')
#
#
# def question_answer():
#     from transformers import pipeline
#
#     oracle = pipeline(model="deepset/roberta-base-squad2")
#
#     oracle(question="Where do I live?", context="My name is Wolfgang and I live in Berlin")
#     {'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}


def pipeline(pdf_path, prompt):

    pdf_path = Path.cwd() / pdf_path

    # First phase of processing the PDF
    pages_list = process_pdf(pdf_path)

    # Second phase of processing the pages of the PDF
    aggregated_docs = []
    for i in range(len(pages_list)):
        chunker = page_chunker(pages_list[i])
        aggregated_docs.extend(chunker)

    filename = f"{pdf_path.stem}"

    filename = filename.replace(" ", "_")
    logger.debug(f"Filename: {filename}")

    index_path_check = Path.cwd() / "vector_storage" / f"{filename}.faiss"
    config_path_check = Path.cwd() / "vector_storage" / f"{filename}.json"

    # Third phase of processing the PDF
    if index_path_check.exists():
        logger.debug(f"Index exists. Attempting loading database.")
        # Load the FAISSDocumentStore if the database already exists
        document_store = FAISSDocumentStore.load(
            index_path=f"vector_storage/{filename}.faiss",
            config_path=f"vector_storage/{filename}.json",
        )
        # TODO:
        # here there's a special case when only the .db file is present, and maybe the index is corrupt or missing
        # if no index file and maybe even the config file, it should regenerate those files somehow
        logger.debug(f"Loading database complete.")

        document_store.write_documents(aggregated_docs, index=filename, duplicate_documents="skip")
        logger.debug("Wrote docs inside document store")

        retriever = get_retriever(document_store)
        logger.debug("Loaded retriever.")

        document_store.update_embeddings(
            retriever=retriever, update_existing_embeddings=False
        )
        logger.debug("Updated embeddings with the help of retriever.")

        # Save the index and configuration
        document_store.save(
            index_path=f"vector_storage/{filename}.faiss",
            config_path=f"vector_storage/{filename}.json",
        )

    else:
        # Otherwise, initialize a new FAISSDocumentStore
        logger.debug(f"Creating database.")
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///vector_storage/faiss-storage.db",
            faiss_index_factory_str="Flat",
            index=filename,
            embedding_dim=1024,
            similarity="cosine",
            embedding_field="meta",
            progress_bar=False
        )
        logger.debug(f"Writing database completed.")

        document_store.write_documents(aggregated_docs, index=filename, duplicate_documents="overwrite")
        logger.debug("Wrote docs inside document store")

        retriever = get_retriever(document_store)
        logger.debug("Loaded retriever.")

        document_store.update_embeddings(
            retriever=retriever, update_existing_embeddings=False
        )
        logger.debug("Updated embeddings with the help of retriever.")

        # Save the index and configuration
        document_store.save(
            index_path=f"vector_storage/{filename}.faiss",
            config_path=f"vector_storage/{filename}.json",
        )

    retrieved_pages = retriever.retrieve(query=f"{prompt}", document_store=document_store, scale_score=False)

    feed_prompt = []
    for result in retrieved_pages:
        logger.debug(f"Score: {round(result.score * 100, 2)}%; File: {result.meta['filename']}, Page: {result.meta['page']}")
        feed_prompt.append({"result_found": result.content, "filename": result.meta['filename'], "page_number": result.meta['page']})
    logger.debug(f"The prompt feed: {feed_prompt}")

    model = Path.home() / "models" / config.llm.model
    llm = Llama(
        model_path=str(model),
        n_ctx=config.llm.context,
        n_threads=config.llm.threads,
        n_gpu_layers=config.llm.gpu_layers,
        temperature=config.llm.temperature,
        stream=config.llm.stream
    )

    prompt_input = f"""
    {prompt}
    """

    prompt_setup = f"""
    <|im_start|>system
    You are a PDF Reader assistant. You will respond to the user prompt with useful information. You are informed about what to answer based on the following context out of the json context: {feed_prompt}<|im_end|>
    <|im_start|>user
    {prompt_input}<|im_end|>
    <|im_start|>assistant
    """

    llm.reset()
    result = llm(prompt=prompt_setup, max_tokens=128) #, stream=True)
    logger.debug(f"LLM Output: {result['choices'][0]['text']}")

    return result['choices'][0]['text']

        # for chunk in result:
        #     print(chunk['choices'][0]['text'], end='')


def create_parser():
    parser = argparse.ArgumentParser(
        prog=f"{sys.argv[0]}",
        description="Chat with your pdf.",
    )
    parser.add_argument("-v", "--version", dest="version", action="version", version=f"%(prog)% {config.version}")

    parser.add_argument("-i", "--input", type=str, required=True, help="path to the pdf", metavar="some pdf")
    parser.add_argument("-p", "--prompt", type=str, help="optional custom prompt", metavar="prompt")

    parser.epilog = f"Bye"

    return parser


def main():
    setup_logging()
    parser = create_parser()
    args = parser.parse_args()

    if args.input:
        process = pipeline(args.input, prompt=args.prompt)
        print(process)
    else:
        parser.print_help()


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    main()
