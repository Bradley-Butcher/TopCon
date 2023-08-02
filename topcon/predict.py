from typing import List
import lmql

def _list_as_string(list):
    return "[" + ', '.join([f'"{item}"' for item in list]) + "]"

@lmql.query
def lmql_topic_proba(input_text: str, topics: str):
    '''lmql
    argmax
        """Article Text: {input_text}
        Question: What is the topic of this article?\n
        Analysis:[ANALYSIS]\n
        Based on this, the overall topic of the message can be considered to be [TOPIC]"""
    from 
        "openai/text-ada-001"
    distribution
        TOPIC in topics
    '''

def GPT_topic_proba(input_text: str, topics: List[str]) -> dict:
    # topics = _list_as_string(topics)
    probs = lmql_topic_proba(input_text, topics=topics).variables["P(TOPIC)"]
    breakpoint()
    return {topic: prob for topic, prob in zip(topics, probs)}