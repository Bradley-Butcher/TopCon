from topcon import APSConformalPredictor, GPT_topic_proba

predictor = APSConformalPredictor.from_hf_datasets(
    hf_repo_name='yahoo_answers_topics',
    topic_column='topic',
    text_columns=['question_title', 'question_content'],
    topic_proba=GPT_topic_proba,
    calibration_size=500,
    save_path='test_data/yahoo_answers_topics',
)

predictor.calibrate(alpha=0.05)