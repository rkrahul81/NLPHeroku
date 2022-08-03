from typing import Optional, Union

import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

class E2EQGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        use_cuda: bool
    ) :

        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"
        
        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
    
    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)
        
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions
    
    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
SUPPORTED_TASKS = {
    "e2e-qg": {
        "impl": E2EQGPipeline,
        "default": {
            "model": "valhalla/t5-base-e2e-qg",
        }
    }
}

def pipeline(
    task: str,
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    ans_model: Optional = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    use_cuda: Optional[bool] = True,
    **kwargs,
):
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]
    
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    
    if task == "e2e-qg":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
    
    else:
        return task_class(model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer, qg_format=qg_format, use_cuda=use_cuda)
    
    #Create an object of pipeline, e2e-qg is End2End Question Generation
nlp = pipeline("e2e-qg")

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
#Initialize answer_model_name with the name of the Question Answering Model
answer_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
answer_tokenizer = AutoTokenizer.from_pretrained(answer_model_name)
answer_model = AutoModelForQuestionAnswering.from_pretrained(answer_model_name)
answer_pipeline = pipeline('question-answering', model=answer_model_name, tokenizer=answer_model_name)

from flask import Flask, jsonify, request
import sys
import json
app = Flask(__name__)  

#For returning the Generated questions when the context is sent through the API
@app.route('/sendQuestionAnswers', methods = ['POST'])
def sendQuestions():
    #preprocessing required since due to the format sent by the application through the API
    
    #data recieved is in the format '[{"code": 200, "text": context1, "blocks": int},{"code": 200, "text": context2, "blocks": int}]'
    data_list = request.get_json()
    data = data_list[0]
    
    #Preprocessing to remove double quotation marks in just the context since that creates an error when converting to a dict
    #done by replacing " initially with @@@ and then replacing it back after dict conversion
    data_jsons = data.split('}')
    
    final_QAs = []
    
    for data_json_text in data_jsons[:-1]:
        data_json_text+="}"
        data_json_ftext = data_json_text.split('"text": "')[0] + '"text": "'+ data_json_text.split('"text": "')[1].split('", "blocks"')[0].replace('"','@@@') + '", "blocks"' + data_json_text.split('"text": "')[1].split('", "blocks"')[1]
        
        #convert the string to json
        data_json = json.loads(data_json_ftext)
        
        #put the double quotes in the context back in place
        data_json["text"] = data_json["text"].replace('@@@', '"')
        context = data_json["text"]
        
        #generate questions for each context
        questions = nlp(context)
        
        answers = []
        for question in questions:
            QA_input = {'question': question, 'context': context}
            answer_res = answer_pipeline(QA_input)
            answer = answer_res['answer']
            answers.append({'question': question, 'answer':answer})
            print(answers)        
        #concatenate questions
        final_QAs = final_QAs + answers
        print(final_QAs)
    return jsonify(final_QAs)

#For returning the Predicted Answers when the context along with the questions is sent through the API
@app.route('/sendAnswers', methods = ['POST'])
def sendAnswers():
    request_data = request.get_json()    
    answers_ret = []
    print(request_data, file=sys.stderr)
    context = request_data[0]
    questions = request_data[1]
    for question in questions:
        QA_input = {'question': question, 'context': context}
        answer_res = answer_pipeline(QA_input)
        ret_answer = answer_res['answer']
        answers_ret.append({"question":question, "answer":ret_answer})
    print(answer_res['answer'], file=sys.stderr)
    with open('newfile.json', 'w') as f:
        json.dump(answer_res['answer'],f)
    return jsonify(answers_ret)

@app.route("/")
def home_view():
        return "<h1>Hello World!</h1>"
        
app.run()