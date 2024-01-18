from transformers import pipeline

def answer_question(context, question):
    # Initialize a pipeline for question-answering
    question_answerer = pipeline("question-answering")

    # Use the pipeline to answer a question
    result = question_answerer(question=question, context=context)

    return result['answer']

# Example context and question
context = """
The Transformers library provides state-of-the-art machine learning models for natural language processing tasks. 
It is developed by Hugging Face and supports models like BERT, GPT-2, T5, and others.
"""

question = "Who develops the Transformers library?"

# Get the answer
answer = answer_question(context, question)
print(f"Question: {question}")
print(f"Answer: {answer}")
