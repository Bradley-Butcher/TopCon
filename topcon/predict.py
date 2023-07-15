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
        Answer:[ANALYSIS]\n
        Based on this, the overall topic of the message can be considered to be [TOPIC]"""
    from 
        "openai/gpt-3.5-turbo"
    distribution
        TOPIC in {topics}
    '''

def topic_proba(input_text: str, topics: List[str]) -> dict:
    return lmql_topic_proba(input_text, topics=topics).variables["P(CLS)"]