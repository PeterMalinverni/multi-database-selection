import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
with open('keys.txt', 'r') as file:
    openai.api_key = file.read().strip()

def load_and_prepare_datasets(dataset_filenames):
    dataset_texts = {}
    for filename in dataset_filenames:
        df = pd.read_csv(filename)
        # Convert the DataFrame to a string representation
        data_str = df.to_string(index=False)
        dataset_texts[filename] = data_str
    return dataset_texts

def get_embedding(text, model="text-embedding-ada-002"):
    # Ensure the text is within the token limit
    text = text.replace("\n", " ")
    client = openai.OpenAI()  # Create client instance
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def select_dataset_ml(question, dataset_texts):
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

def get_llm_response(question, dataset_filename, dataset_text):
    # Create the prompt
    client = openai.OpenAI()  # Create client instance
    prompt = f"""
You are an AI assistant with access to the following dataset:

{dataset_text}

Based on this data, please answer the following question:

"{question}"
"""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes data and answers questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7,
    )

    # Extract the response text
    answer = response.choices[0].message.content.strip()
    return answer

# Example usage
if __name__ == "__main__":
    user_question = input("Please enter your question: ")

    # Load and prepare datasets
    dataset_filenames = ['shopping_habits.csv', 'weekly_searches_for_programming_languages.csv']
    dataset_texts = load_and_prepare_datasets(dataset_filenames)

    # Select the most relevant dataset using ML
    selected_dataset = select_dataset_ml(user_question, dataset_texts)
    print(f"Selected Dataset: {selected_dataset}")

    # Get the text of the selected dataset
    selected_dataset_text = dataset_texts[selected_dataset]

    # Get the LLM response
    answer = get_llm_response(user_question, selected_dataset, selected_dataset_text)
    print("\nLLM Response:")
    print(answer)
