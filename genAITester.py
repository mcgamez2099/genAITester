import chainlit as cl
from chainlit.input_widget import Select, Slider

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import subprocess, platform, base64, io, os
from PIL import Image



# get current available models
models=ChatNVIDIA.get_available_models()
availableModels=[]
for model in models:
    # save all founds models in availableModels
    availableModels.append(model.id)
# set llama3-8b-instuct as default model
modelIndex=availableModels.index("meta/llama3-8b-instruct")
currentModel=availableModels[modelIndex]
multimodalList = ["nvidia/neva-22b","microsoft/kosmos-2", "adept/fuyu-8b"]
# text_splitter for vector databes 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# sentence tranformer embedding class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self._embedding_function = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        embeddings = self._embedding_function.encode(texts, convert_to_numpy=True).tolist()
        return [list(map(float, e)) for e in embeddings]
    def embed_query(self, text):
        embeddings = self._embedding_function.encode([text], convert_to_numpy=True).tolist()
        return [list(map(float, e)) for e in embeddings][0]
embeddings = SentenceTransformerEmbeddings()


# prompts 
chat_prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful and honest AI assistant. Always answer as helpfully as possible, while being safe. If you don't know the answer to a question, please don't share false information"), ("user", "{input}")])

qa_prompt = ChatPromptTemplate.from_messages([("human", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Question: {question}  Context: {context}  Answer: ")])

pdf_qa_template1 = """Use the following pieces of context to answer the users question. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    QUESTION: {question}
    =========
    {summaries}
    =========
    ANSWER:
    """

# load website and save it to vector database
def loadWebsite2VectorDB(webUrl):
    loader = WebBaseLoader(webUrl)
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=NVIDIAEmbeddings(model="NV-Embed-QA"))
    return vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# chat history 
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)


# load chromaDB and collection
def make_doc_chain(collection_name,k_value):
    chromaPersist_directory = './src/chromaDB'
    if os.path.isdir(chromaPersist_directory):
        vector_store = Chroma(collection_name=collection_name,persist_directory=chromaPersist_directory,embedding_function=embeddings)
        memory.clear()
        pdf_qa_prompt = PromptTemplate(template=pdf_qa_template1, input_variables=["summaries", "question"])
        pdf_chain_type_kwargs = {"prompt": pdf_qa_prompt}
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatNVIDIA(model=currentModel),
            chain_type="stuff",
            chain_type_kwargs=pdf_chain_type_kwargs,
            retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={'k': k_value}),
            return_source_documents=True,
            memory=memory,
            verbose=True,     
        )
    else:
        return "noVDB"


# convert img to base64
def image_to_bytes_data_uri(img):
    base64_data = base64.b64encode(img).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"

# DuckDuckGo Search Tool
wrapper = DuckDuckGoSearchAPIWrapper(max_results=6)
ddg_search = DuckDuckGoSearchResults(api_wrapper=wrapper)
ddg_sources = []
@tool
def duckDuckGoSearchWithSources(question: str) -> int:
    """duckDuckGoSearch Tool"""
    res = ddg_search.run(question)
    for link in res.split("], ["):
        if link:
            link = link.split(", link: ")[1].strip()
            ddg_sources.append(link)
    return res


# default app settings
prompt = chat_prompt
temperature=0.5
max_tokens=1000
top_p=0.9
k_value=6
DuckDuckGoAgentMaxIterations=2
vectorDB_available=True
chatMode="Chat"
# 

def loadLLM():
    return ChatNVIDIA(model=currentModel,temperature=temperature,max_tokens=max_tokens,top_p=top_p)

@cl.on_chat_start
async def on_chat_start():

    if os.getenv("NVIDIA_API_KEY") is None:
        await cl.Message(content=f"NVIDIA_API_KEY is not set as env variable\nPlease setup your api key to use this app").send()
    else:
        global modelIndex
        # fall back if llama3 wasn't found
        if not isinstance(modelIndex, int):
            modelIndex=0     
        settings = await cl.ChatSettings(
            [
                Select(
                    id="Model", label="LLM: Model",
                    values=availableModels,
                    initial_index=modelIndex,
                ),
                Select(
                    id="LLM Mode", label="LLM: Mode",
                    values=["Chat", "ChatWithHistory", "WebsiteChat", "vdbChat", "createVDB", "txtFileChat", "imgChat", "DuckDuckGoSearch"],
                    initial_index=0,
                ),
                Select(
                    id="vectorDB CollectionName", label="vectorDB: collectionName",
                    values=["Books-Collection","PDF-Collection"],
                    initial_index=0,
                ),
                Slider(
                    id="Temperature", label="LLM: Temperature",
                    initial=temperature,
                    min=0, max=1, step=0.1,
                ),
                Slider(
                    id="maxTokens", label="LLM: max_tokens",
                    initial=max_tokens,
                    min=0, max=8000, step=1,
                ),
                Slider(
                    id="top_p", label="LLM: top_p",
                    initial=top_p,
                    min=0, max=1, step=0.1,
                ),
                Slider(
                    id="k_value", label="vectorDB: k_value (number of results to retrieve)",
                    initial=k_value,
                    min=0, max=12, step=1,
                ),
                Slider(
                    id="DuckDuckGoAgentMaxIterations", label="DuckDuckGoAgent: maxInterations",
                    initial=DuckDuckGoAgentMaxIterations,
                    min=0, max=20, step=1,
               ),
            ]
        ).send()
        await setup_agent(settings)

       
 

@cl.action_callback("start")
async def on_action(action):
    # action button to load the pdf2chromadb.py script
    global chatMode
    await cl.Message(content=f"creating database please wait").send()
    if platform.system() == "Darwin" or platform.system() == "Linux":
        subprocess.run(["python", "pdf2chromadb.py"])
    else:
        subprocess.run("python pdf2chromadb.py", shell=True, check=True)
    await action.remove()
    pdf_chain = make_doc_chain("PDF-Collection",k_value)   
    await cl.Message(content=f"vector database with collection 'PDF-Collection' created succesfully").send()
    cl.user_session.set("chain", pdf_chain)
    await cl.Message(content=f"You can now ask questions!").send()
    # change the changeMode
    chatMode="vdbChat"





@cl.on_settings_update
async def setup_agent(settings):
    global currentModel, prompt, chatMode, modelIndex, temperature, max_tokens, top_p, k_value
    print("on_settings_update", settings)
    chatMode=settings["LLM Mode"]
    currentModel=settings["Model"]
    modelIndex=settings["Model"].index
    temperature=settings["Temperature"]
    max_tokens=settings["maxTokens"]
    top_p=settings["top_p"]
    k_value=settings["k_value"]

    match chatMode: 
        case "Chat":
            await cl.Message(content=f"LLM Chat Mode\n\nYou can use diffent models\ncurrent model: {currentModel}\n\nYou can now ask questions!").send()
            prompt = chat_prompt
            runnable = prompt | loadLLM() | StrOutputParser()
            cl.user_session.set("runnable", runnable)
        case "ChatWithHistory":
            await cl.Message(content=f"LLM Chat Mode with history\n\ncurrent model: {currentModel}\n\nYou can now ask questions!").send()
            chat = loadLLM()
            conversation = ConversationChain(llm=chat, memory=ConversationBufferMemory())
            cl.user_session.set("conversation", conversation)
        case "WebsiteChat":
            prompt = qa_prompt
            print("load websiteChat")
            res = await cl.AskUserMessage(content="LLM Website Chat\nPlease enter a website url:").send()
            if res:
                await cl.Message(content=f"Start to chunk the webite: {res['output']}, please wait").send()
                startChunking=True
            if startChunking == True:
                await cl.sleep(1)
                retriever=loadWebsite2VectorDB(res['output'])
                await cl.Message(content="Start with your questions", author="Chatbot").send()
            runnable = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | ChatNVIDIA(model=currentModel)
                | StrOutputParser()
            )
            cl.user_session.set("runnable", runnable)
        case "vdbChat":
            await cl.Message(content=f"LLM VectorDatabase Chat\nloading vectorDB...").send()
            pdf_chain = make_doc_chain(settings["vectorDB CollectionName"],k_value)     
            if pdf_chain == "noVDB":
                await cl.Message(content=f"No VectorDatabase found!\nPlease change to createVDB-Mode to create a chromaVDB with own pdf files").send()
            else:    
                await cl.Message(content=f"loading vectorDB done.").send()
                cl.user_session.set("chain", pdf_chain)
                await cl.Message(content=f"You can now ask questions!").send()
        case "createVDB":
            await cl.Message(content="Insert PDF Documents in your ./pdfs folder", author="Chatbot").send()
            actions = [
                cl.Action(name="start", value="start", description="start")
            ]
            await cl.Message(content="Press Start to load your documents to vector database", actions=actions).send()
        case "txtFileChat":
            await cl.Message(content=f"LLM txtFile Chat").send()
            files = None
            # Wait for the user to upload a file
            while files == None:
                files = await cl.AskFileMessage(
                    content="Please upload a text file to begin!",
                    accept=["text/plain"],
                    max_size_mb=20,
                    timeout=180,
                ).send()
            file = files[0]
            await cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True).send()
            with open(file.path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split the text into chunks
            texts = text_splitter.split_text(text)
            # Create a metadata for each chunk
            metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
            # Create a Chroma vector store
            docsearch = await cl.make_async(Chroma.from_texts)(
                texts, embeddings, metadatas=metadatas
            )
            # Create a chain that uses the Chroma vector store
            chain = ConversationalRetrievalChain.from_llm(
                loadLLM(),
                chain_type="stuff",
                retriever=docsearch.as_retriever(),
                memory=memory,
                return_source_documents=True,
            )
            await cl.Message(content=f"Processing `{file.name}` done. You can now ask questions!").send()
            cl.user_session.set("chain", chain)
        case "imgChat":
            if not currentModel in multimodalList:
                await cl.Message(content=f"No compatibel model loaded\nYou can use: {multimodalList}").send()
                modelIndex=availableModels.index("nvidia/neva-22b")
                currentModel=availableModels[modelIndex]
                await cl.Message(content=f"Changed model to nvidia/neva-22b").send()
            await cl.Message(content=f"Please upload an image (png, jpg and jpeg) and type in your question").send()
        case "DuckDuckGoSearch":
            tools = [
                Tool(
                name="duckDuckGoSearchWithSources",
                func=duckDuckGoSearchWithSources.run,
                description="Useful to browse information from the Internet.",
                ) 
             ]
            agent = initialize_agent(
                tools, llm=loadLLM(), agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, 
                handle_parsing_errors=True,early_stopping_method='generate', max_iterations=DuckDuckGoAgentMaxIterations
            )
            cl.user_session.set("agent", agent)
            await cl.Message(content=f"DuckDuckGo Search Engine loaded.\nYou can now ask questions!").send()
          

@cl.on_message
async def on_message(message: cl.Message): 
    match chatMode:
        case "Chat":
            msg = cl.Message(content="")
            runnable = cl.user_session.get("runnable")  
            for chunk in await cl.make_async(runnable.stream)(
                {"input": message.content},
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
            ):
                await msg.stream_token(chunk)
            await msg.send()

        case "ChatWithHistory":
            conversation = cl.user_session.get("conversation")
            cb = cl.LangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["Final","Answer"])
            cb.answer_reached = True
            res = await conversation.acall(message.content, callbacks=[cb])
            await cl.Message(content=res["response"]).send() 

        case "WebsiteChat":
            runnable = cl.user_session.get("runnable")  
            answer = runnable.invoke(message.content)
            await cl.Message(content=answer).send()

        case "vdbChat":
            chain = cl.user_session.get("chain")
            cb = cl.LangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["Assistant"])
            cb.answer_reached = True
            res = await cl.make_async(chain)({"question": message.content}, callbacks=[cb])
            answer = res["answer"]
            await cl.Message(content=answer).send()
            source = res["source_documents"]
            text_elements = []
            text_titles = []
            sources = []
            for document in source:
                page_number=f"Page: {document.metadata['page_number']}"
                title=f"Title: {document.metadata['title']}"
                text_chunk=f"{title}\n\n {document.page_content[:250]}...\n"
                text_elements.append(cl.Text(content=text_chunk, name=page_number))
                text_titles.append(cl.Text(content=title, name=title))

            source_names = [text_el.name for text_el in text_elements]
            source_title = [text_el.name for text_el in text_titles]
            if source_names:
                sources = f"\n\nSources: {', '.join(source_names)}"
            else:
                sources = "\nNo sources found"
            await cl.Message(content=sources, elements=text_elements).send()

        case "txtFileChat":
            chain = cl.user_session.get("chain") 
            cb = cl.AsyncLangchainCallbackHandler()
            res = await chain.acall(message.content, callbacks=[cb])
            answer = res["answer"]
            source_documents = res["source_documents"]  
            text_elements = []  
            if source_documents:
                for source_idx, source_doc in enumerate(source_documents):
                    source_name = f"source_{source_idx}"
                    # Create the text element referenced in the message
                    text_elements.append(
                        cl.Text(content=source_doc.page_content, name=source_name)
                    )
                source_names = [text_el.name for text_el in text_elements]

                if source_names:
                    answer += f"\nSources: {', '.join(source_names)}"
                else:
                    answer += "\nNo sources found"
            await cl.Message(content=answer, elements=text_elements).send()
            
        case "imgChat":
            if not message.elements:
                await cl.Message(content="No file attached").send()
                return
            images = [file for file in message.elements if "image" in file.mime]
            await cl.Message(content=f"Received {len(images)} image(s)").send()
            
            imageMineType=images[0].mime       
            image = Image.open(images[0].path)
            # resize image for llm
            image_resize = image.resize((250,250))
            image_resize.save(f"./img1.{imageMineType.split('/')[1]}")
            with open(f"./img1.{imageMineType.split('/')[1]}", "rb") as f:
                imgBytes = f.read()
            # convert the resized image
            data_uri = image_to_bytes_data_uri(imgBytes)
            llm = loadLLM()
            result = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": message.content },
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]
                )
            ])
            await cl.Message(content=result.content).send()

        case "DuckDuckGoSearch":
            global ddg_sources
            agent = cl.user_session.get("agent")
            response = await agent.acall({"input": message.content}, callbacks=[cl.LangchainCallbackHandler()])
            answer = response["output"]
            source_link_url = []  
            if ddg_sources:
                for link in ddg_sources:
                    source_link_url.append(link)
            ddg_sources = []
            if source_link_url:
                answer += f"\n\nSources: \n{', '.join(source_link_url)}\n"
            await cl.Message(content=answer).send()
            


