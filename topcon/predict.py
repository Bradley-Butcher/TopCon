from typing import List
import lmql

def _list_as_string(list):
    return "[" + ', '.join(list) + "]"

@lmql.query
def lmql_topic_proba(input_text: str, topics: List[str]):
    topics = _list_as_string(topics)
    '''lmql
    argmax
    """Article Text: {input_text}
    Question: What is the topic of this article?\n
    A:[ANALYSIS]\n
    Based on this, the overall sentiment of the message can be considered to be [CLASSIFICATION]"""
    from 
        "openai/gpt-3.5-turbo"
    distribution
        CLASSIFICATION in {topics}
    '''

def topic_proba(input_text: str, topics: List[str]) -> dict:
    return lmql_topic_proba(input_text, topics=topics).variables["P(CLS)"]