import streamlit as st
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from fuzzywuzzy import fuzz
import re
import numpy as np

# Load course data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("free_course.csv")
        required_columns = ["Course Name", "Description", "Course URL", "Lessons", "Curriculum"]

        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must include the columns: {', '.join(required_columns)}")
            st.stop()

        df["Lessons"] = df["Lessons"].apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else 0
        )
        df["Lessons"] = df["Lessons"].astype(int)
        df["Lessons"] = df["Lessons"].apply(lambda x: max(0, x))
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Create vectorstore
@st.cache_resource
def create_vectorstore(df):
    try:
        documents = [
            Document(
                page_content=str(row["Description"]) + " " + str(row["Curriculum"]),
                metadata={
                    "title": row["Course Name"],
                    "url": row["Course URL"],
                    "lessons": row["Lessons"],
                },
            )
            for _, row in df.iterrows()
        ]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        st.stop()

# Load Hugging Face LLM
@st.cache_resource
def load_huggingface_llm():
    try:
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        return llm
    except Exception as e:
        st.error(f"Error loading Hugging Face LLM: {e}")
        st.stop()

# Calculate unified score
def calculate_unified_score(query, vectorstore, df, semantic_weight=0.85, fuzzy_weight=0.25):
    # Semantic similarity scores
    retriever = vectorstore.as_retriever()
    results = retriever.get_relevant_documents(query)
    semantic_scores = {doc.metadata["title"]: doc.metadata["lessons"] for doc in results}

    # Fuzzy matching scores
    df["fuzzy_score"] = df["Course Name"].apply(
        lambda x: fuzz.partial_ratio(query.lower(), str(x).lower())
    )

    # Popularity (Assuming you have a column like 'Popularity' in your dataset)
    # df["popularity_score"] = df["Popularity"].apply(lambda x: min(1, x / df["Popularity"].max()))

    # Normalize scores
    max_semantic = max(semantic_scores.values(), default=1)
    max_fuzzy = df["fuzzy_score"].max()

    df["semantic_score"] = df["Course Name"].map(lambda x: semantic_scores.get(x, 0) / max_semantic)
    df["fuzzy_score"] = df["fuzzy_score"] / max_fuzzy

    # Weighted unified score
    df["unified_score"] = (
        semantic_weight * df["semantic_score"] +
        fuzzy_weight * df["fuzzy_score"]
        # popularity_weight * df["popularity_score"]
    )

    return df.sort_values(by="unified_score", ascending=False)



# Main
def main():
    st.title("Intelligent Course Search Tool")
    st.write("Find the most relevant free courses using advanced search techniques.")

    df = load_data()
    vectorstore = create_vectorstore(df)

    # Advanced filters
    num_courses = st.slider("Number of suggestions:", 1, 10, 5)
    lesson_min, lesson_max = st.slider(
        "Lesson Range (min-max):", min_value=10, max_value=400, value=(10, 400)
    )

    query = st.text_input("Enter your search query:", "")
    if query:
        with st.spinner("Searching..."):
            try:
                # Filter courses based on lesson range
                filtered_courses = df[
                    (df["Lessons"] >= lesson_min) & (df["Lessons"] <= lesson_max)
                ]

                # Calculate unified scores
                ranked_courses = calculate_unified_score(query, vectorstore, filtered_courses)

                # Display results
                st.subheader(f"Top {num_courses} Relevant Courses")
                for _, row in ranked_courses.head(num_courses).iterrows():
                    st.markdown(f"**[{row['Course Name']}]({row['Course URL']})**")
                    st.write(f"Lessons: {row['Lessons']}")
                    st.write(f"Unified Score: {row['unified_score']:.2f}")
                    st.write("------")
            except Exception as e:
                st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
