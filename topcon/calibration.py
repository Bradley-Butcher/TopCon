from typing import List, Tuple
import numpy as np
import datasets
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

class ConformalPredictor:
    def __init__(self, calibration_set: List[Tuple[str, str]], topic_proba, topics: List[str], save_path: Path = None):
        self.topic_proba = topic_proba
        self.topics = topics
        self.calibration_set = calibration_set
        self.save_path = save_path
        self.nonconformity_scores = self.get_calibration_distribution()
        self.qhat = None

    def get_calibration_distribution(self):
        # calculate nonconformity scores
        nonconformity_scores = []
        for text, true_topic in self.calibration_set:
            proba = self.topic_proba(text, self.topics)
            true_proba = proba[true_topic]
            nonconformity_score = 1 - true_proba
            nonconformity_scores.append(nonconformity_score)
        return np.array(nonconformity_scores)

    def get_adjusted_quantile(self, alpha=0.1, adjust: bool = True):
        n = len(self.nonconformity_scores)
        if adjust:
            q_level = np.min([1, np.ceil((n+1)*(1-alpha))/n])
        else:
            q_level = 1-alpha
        self.qhat = np.quantile(self.nonconformity_scores, q_level, interpolation='higher')
        if self.save_path:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self.qhat, f)
        return self.qhat

    def get_prediction_sets(self, new_text: str):
        proba = self.topic_proba(new_text, self.topics)
        prediction_sets = proba >= (1-self.qhat)
        return prediction_sets

    def plot_nonconformity_distribution(self):
        plt.hist(self.nonconformity_scores, bins=30, alpha=0.5)
        plt.axvline(x=self.qhat, color='r', linestyle='--')
        plt.title('Nonconformity Score Distribution')
        plt.xlabel('Nonconformity Score')
        plt.ylabel('Frequency')
        plt.show()

    @classmethod
    def from_hf_datasets(
        cls, 
        hf_repo_name: str, 
        topic_column: str, 
        text_columns: List[str],
        topic_proba: callable,
        calibration_size: int,
        save_path: Path = None
    ):
        # load data from huggingface datasets
        dataset = datasets.load_dataset(hf_repo_name)
        topics = dataset['train'].features[topic_column].names
        calibration_set = dataset['train'].sample(n=calibration_size)
        calibration_set.add_column(
            'combined_text',
            [
                ' '.join([row[text_columns[tc]] for tc in text_columns])
                for row in calibration_set
            ]
        )
        calibration_set = calibration_set['combined_text', topic_column]
        return cls(calibration_set, topic_proba, topics, save_path)

    @classmethod
    def load(cls, load_path: Path):
        with open(load_path, 'rb') as f:
            qhat = pickle.load(f)
        return qhat
