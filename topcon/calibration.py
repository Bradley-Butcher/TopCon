from typing import List, Tuple
import numpy as np
import datasets
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod
from loguru import logger
from tqdm import tqdm
class BaseConformalPredictor(ABC):
    def __init__(
        self, 
        calibration_set: List[Tuple[str, str]], 
        topic_proba, 
        topics: List[str], 
        save_path: Path = None, 
        save_interval: int = 10
    ):
        self.topic_proba = topic_proba
        self.topics = topics
        self.calibration_set = calibration_set
        self.save_path = Path(save_path)
        self.save_interval = save_interval
        self.nonconformity_scores = self.load_nonconformity_scores() if self.save_path and self.save_path.exists() else None
        self.qhat = None


    def save_nonconformity_scores(self):
        logger.info(f'Saving nonconformity scores to {self.save_path / "scores.pkl"}')
        if self.save_path:
            with open(self.save_path / "scores.pkl", 'wb') as f:
                pickle.dump(self.nonconformity_scores, f)

    def load_nonconformity_scores(self):
        logger.info(f'Loading nonconformity scores from {self.save_path / "scores.pkl"}')
        with open(self.save_path / "scores.pkl", 'rb') as f:
            return pickle.load(f)

    def plot_nonconformity_distribution(self):
        assert self.nonconformity_scores is not None, "Nonconformity scores not calculated yet"
        assert self.qhat is not None, "Q-hat not calculated yet"
        plt.hist(self.nonconformity_scores, bins=30, alpha=0.5)
        plt.axvline(x=self.qhat, color='r', linestyle='--')
        plt.title('Nonconformity Score Distribution')
        plt.xlabel('Nonconformity Score')
        plt.ylabel('Frequency')
        plt.show()

    @abstractmethod
    def get_calibration_distribution(self):
        pass

    @abstractmethod
    def get_prediction_sets(self, new_text: str):
        pass

    @abstractmethod
    def get_adjusted_quantile(self):
        pass

    def calibrate(self, alpha: float = 0.1):
        self.get_calibration_distribution()
        self.get_adjusted_quantile(alpha=alpha)

    def predict(self, new_text: str):
        prediction_sets = self.get_prediction_sets(new_text)
        return prediction_sets

    @classmethod
    def load_from_qhat(cls, load_path: Path):
        logger.info(f'Loading ConformalPredictor from {load_path / "qhat.pkl"}')
        with open(load_path / "qhat.pkl", 'rb') as f:
            qhat = pickle.load(f)
        instance = cls(None, None, None, None, None)
        instance.qhat = qhat
        return instance

    @classmethod
    def load_from_scores(
        cls, 
        load_path: Path,
        ):
        logger.info(f'Loading ConformalPredictor from {load_path / "scores.pkl"}')
        with open(load_path / "scores.pkl", 'rb') as f:
            nonconformity_scores = pickle.load(f)
        instance = cls(None, None, None, None, None)
        instance.nonconformity_scores = nonconformity_scores
        return instance

    def get_adjusted_quantile(self, alpha=0.1, adjust: bool = True):
        n = len(self.nonconformity_scores)
        if adjust:
            q_level = np.min([1, np.ceil((n+1)*(1-alpha))/n])
        else:
            q_level = 1-alpha
        self.qhat = np.quantile(self.nonconformity_scores, q_level, interpolation='higher')
        if self.save_path:
            with open(self.save_path / "qhat.pkl", 'wb') as f:
                pickle.dump(self.qhat, f)
        return self.qhat

    @classmethod
    def from_hf_datasets(
        cls, 
        hf_repo_name: str, 
        topic_column: str, 
        text_columns: List[str],
        topic_proba: callable,
        calibration_size: int,
        save_path: Path = None,
        random_seed: int = 42,
        save_interval: int = 10
    ):
        # Set seed for reproducibility
        np.random.seed(random_seed)

        # load data from huggingface datasets
        dataset = datasets.load_dataset(hf_repo_name)
        topics = dataset['train'].features[topic_column].names
        calibration_set = dataset['train'].shuffle(seed=random_seed).select(range(calibration_size))
        calibration_set = calibration_set.add_column(
            'combined_text',
            [
                ' '.join([row[tc] for tc in text_columns])
                for row in calibration_set
            ]
        )
        columns_to_keep = ['combined_text', topic_column]
        columns_to_remove = [col for col in calibration_set.column_names if col not in columns_to_keep]
        calibration_set = calibration_set.remove_columns(columns_to_remove)
        return cls(calibration_set, topic_proba, topics, save_path, save_interval)

class ConformalPredictor(BaseConformalPredictor):
    def __init__(
        self, 
        calibration_set: List[Tuple[str, str]], 
        topic_proba, 
        topics: List[str], 
        save_path: Path = None,
        save_interval: int = 10
    ):
        super().__init__(calibration_set, topic_proba, topics, save_path, save_interval)

    def get_calibration_distribution(self):
        logger.info("Calculating calibration distribution")
        # calculate nonconformity scores
        nonconformity_scores = []
        for idx, (text, true_topic) in tqdm(enumerate(self.calibration_set), total=len(self.calibration_set)):
            proba = self.topic_proba(text, self.topics)
            true_proba = proba[true_topic]
            nonconformity_score = 1 - true_proba
            nonconformity_scores.append(nonconformity_score)
            if idx % self.save_interval == 0:
                self.save_nonconformity_scores()
        self.nonconformity_scores = np.array(nonconformity_scores)
        return self.nonconformity_scores

    def get_prediction_sets(self, new_text: str):
        proba = self.topic_proba(new_text, self.topics)
        prediction_sets = proba >= (1-self.qhat)
        return prediction_sets

class APSConformalPredictor(BaseConformalPredictor):

    def __init__(self, calibration_set: List[Tuple[str, str]], topic_proba, topics: List[str], save_path: Path = None, save_interval: int = 10):
        super().__init__(calibration_set, topic_proba, topics, save_path, save_interval)

    def get_calibration_distribution(self):
        logger.info("Calculating calibration distribution for APS")
        nonconformity_scores = []
        for text, true_topic in self.calibration_set:
            proba = self.topic_proba(text, self.topics)
            sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)
            cumulative_proba = 0
            for topic, prob in sorted_proba:
                cumulative_proba += prob
                if topic == true_topic:
                    break
            nonconformity_score = 1 - cumulative_proba
            nonconformity_scores.append(nonconformity_score)
        return np.array(nonconformity_scores)

    def get_prediction_sets(self, new_text: str):
        proba = self.topic_proba(new_text, self.topics)
        sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)
        prediction_sets = []
        cumulative_proba = 0
        for topic, prob in sorted_proba:
            cumulative_proba += prob
            if cumulative_proba > self.qhat:
                break
            prediction_sets.append(topic)
        return prediction_sets