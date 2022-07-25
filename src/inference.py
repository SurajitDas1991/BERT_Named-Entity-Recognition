from transformers import pipeline

class InferenceModel:
    def __init__(self,model,tokenizer,sentence) -> None:
        self.model=model
        self.tokenizer=tokenizer
        self.sentence=sentence

    def create_pipeline(self):
        self.pipe = pipeline(task="token-classification", model=self.model.to("cpu"), tokenizer=self.tokenizer, aggregation_strategy="simple")

    def perform_inference(self):
        print(self.pipe(self.sentence))
