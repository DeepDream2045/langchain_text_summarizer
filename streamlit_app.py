import os, streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate


# Streamlit app
st.subheader('Text Summarization using LangChain')

# Get OpenAI API key and source text input
openai_api_key = ""
source_text = st.text_area("Copy the text you want summarize", height=300)

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not source_text.strip():
        st.error(f"Please provide the the text.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Split the source text
                text_splitter = CharacterTextSplitter()
                texts = text_splitter.split_text(source_text)

                # Create Document objects for the texts
                docs = [Document(page_content=t) for t in texts]#[:3]]

                # Initialize the OpenAI module, load and run the summarize chain
                # llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


                # question_prompt_template = """
                #                   Please provide a summary of the following text.
                #                   TEXT: {text}
                #                   SUMMARY:
                #                   """
                # question_prompt = PromptTemplate(
                #     template=question_prompt_template, input_variables=["text"]
                # )
                #
                # refine_prompt_template = """
                #               Write a concise summary of the following text delimited by triple backquotes.
                #               Return your response in bullet points which covers the key points of the text.
                #               ```{text}```
                #               BULLET POINT SUMMARY:
                #               """
                #
                # refine_prompt = PromptTemplate(
                #     template=refine_prompt_template, input_variables=["text"]
                # )
                # chain = load_summarize_chain(
                #                                 llm,
                #                                 chain_type="refine",
                #                                 question_prompt=question_prompt,
                #                                 refine_prompt=refine_prompt,
                #                                 return_intermediate_steps=True,
                #                             )


                # Initialize the OpenAI module, load and run the summarize chain
                llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k', openai_api_key=openai_api_key)
                chain = load_summarize_chain(llm, chain_type="refine")
                summary = chain.run(docs)

                # Display summary
                st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
