import openai
import pandas as pd
import numpy as np
from typing import List, Dict

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
from sklearn.metrics.pairwise import cosine_similarity 

"""
A data analysis assistant that uses OpenAI's GPT-4 and embeddings to intelligently select 
and analyze datasets based on natural language questions.
"""

def select_relevant_dataset(question: str, dataset_texts: Dict[str, str]) -> str:
    """
    This is the core of the assignment.
    Select the most relevant dataset for a given question using embeddings similarity.
    
    Args:
        question (str): User's natural language question
        dataset_texts (dict): Mapping of dataset filenames to their text content
        
    Returns:
        str: Filename of the most relevant dataset
    """
    # Get embedding for the question
    question_embedding = get_embedding(question)

    similarities = {}
    for filename, text in dataset_texts.items():
        # Get embedding for the dataset text
        dataset_embedding = get_embedding(text)
        # Compute cosine similarity
        similarity = cosine_similarity(
            [question_embedding], [dataset_embedding]
        )[0][0]
        similarities[filename] = similarity

    # Select the dataset with the highest similarity
    selected_dataset = max(similarities, key=similarities.get)
    return selected_dataset

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Get OpenAI embeddings for a given text.

    Embeddings are a numerical representation of the text 
    that can be used for similarity comparison.
    
    Args:
        text (str): Text to generate embeddings for
        model (str): OpenAI embedding model to use
        
    Returns:
        list: Numerical embedding vector
    """
    # Ensure the text is within the token limit
    text = text.replace("\n", " ")
    client = openai.OpenAI()  # Create client instance
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def load_and_prepare_datasets(dataset_filenames: List[str]) -> Dict[str, str]:
    """
    Load multiple CSV datasets and convert them to string representations.
    
    Args:
        dataset_filenames (list): List of CSV filenames to load
        
    Returns:
        dict: Mapping of filenames to their string representations
    """
    dataset_texts = {}
    for filename in dataset_filenames:
        df = pd.read_csv(filename)
        # Convert the DataFrame to a string representation
        data_str = df.to_string(index=False)
        dataset_texts[filename] = data_str
    return dataset_texts

def get_llm_response(question: str, dataset_filename: str, dataset_text: str) -> str:
    """
    Generate an answer to the user's question based on the selected dataset using GPT-4.
    
    Args:
        question (str): User's natural language question
        dataset_filename (str): Name of the selected dataset file
        dataset_text (str): Text content of the selected dataset
        
    Returns:
        str: Generated answer from GPT-4
    """
    # Create the prompt
    client = openai.OpenAI()  # Create client instance
    prompt = (
        "You are an AI assistant with access to the following dataset:\n\n"
        f"{dataset_text}\n\n"
        "Based on this data, please answer the following question:\n\n"
        f"\"{question}\""
    )

    # Call the OpenAI API
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes data and concisely answers questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7,
    )

    # Extract the response text
    answer = response.choices[0].message.content.strip()
    return answer


if __name__ == "__main__":
    # Set your OpenAI API key
    with open('keys.txt', 'r') as file:
        openai.api_key = file.read().strip()

    user_question = input("Please enter your question: ")

    # Load and prepare datasets
    dataset_filenames = ['shopping_habits.csv', 'weekly_searches_for_programming_languages.csv']
    dataset_texts = load_and_prepare_datasets(dataset_filenames)

    # Select the most relevant dataset using ML
    selected_dataset = select_relevant_dataset(user_question, dataset_texts)
    print(f"Selected Dataset: {selected_dataset}")

    # Get the text of the selected dataset
    selected_dataset_text = dataset_texts[selected_dataset]

    # Get the LLM response
    answer = get_llm_response(user_question, selected_dataset, selected_dataset_text)
    print("\nLLM Response:")
    print(answer)
