# Conformal Topic Classification - TopCon

Creating this repository because I have the feeling GPT3.5 with constrained generation, and some calibration with conformal prediction could do very well on topic classification.

Either way, it's generic so you can use whatever topic classification you want.

I called it topcon because it's 11pm, I'm tired, and lacking imagination apparently.

## Installation

Install with poetry:

```bash
pip install poetry
poetry install
```

## Usage

Requires calibrating via a calibration set before prediction. There is a huggingface datasets class-method for ease of use. Below is an example.

```python
from topcon.predict import topic_proba
from topcon.calibration import ConformalPredictor

topic_conformer = ConformalPredictor.from_hf_datasets(
    hf_repo_name='yahoo_answers_topics',
    topic_column='topic',
    text_columns=['question_title', 'question_content'],
    topic_proba=topic_proba, #the topic classification function
    calibration_size=1000,
    save_path='save.pkl',
)
```

You can then predict on new text:

```python
get_prediction_sets(
    text='I think football is really amazing because man kick ball good',
)
```

Can also load a previously calibrated model:

```python
topic_conformer = ConformalPredictor.load('save.pkl')
```

## TODO

- [ ] Add Tests
- [ ] Perform extensive evaluation
- [ ] Get a job at DeepMind
