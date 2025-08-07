import argparse
from typing import Optional
import modal
from pathlib import Path
import datetime  # Add this import for timestamp

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
DATASET_DIR = Path("/dataset")
experiments_result_volume = modal.Volume.from_name("experiment-results-vol", create_if_missing=True)
EXPERIMENTS_RESULT_DIR = Path("/experiment_results")

base_image = (
        modal.Image.debian_slim()
        .pip_install("torch", "transformers", "datasets", "evaluate", "tqdm", "huggingface_hub", "hf_transfer", "bert_score", "accelerate", 
                     "rouge_score", "nltk", "peft", "scipy", "scikit-learn", "sentencepiece")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    )

app = modal.App()

@app.function(
    volumes={MODEL_DIR.as_posix(): volume, DATASET_DIR.as_posix(): dataset_volume, 
             EXPERIMENTS_RESULT_DIR.as_posix(): experiments_result_volume
             },
    timeout=60000,
    image=base_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu = "A100-80GB", 
    cpu = 16.0,
    memory= 32768, 
)
async def evaluate_model(model_id : str = "meta-llama/Llama-2-13b-hf", dataset_name: Optional[str] = "knkarthick/dialogsum",
                         batch_size: int = 20, split_name: str = "test", use_peft: bool = False):
    import torch
    import os
    from peft import AutoPeftModelForCausalLM
    import logging
    from tqdm import tqdm
    from datasets import load_dataset, load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import evaluate
    from torch.utils.data import DataLoader 
    from datasets import load_from_disk
    import json
    from itertools import chain
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    bertscore = evaluate.load("bertscore")
    rouge_metric = evaluate.load('rouge')
    
    DATASET_CONTEXT = {
        "summary_dialogue": "You're an expert at summarizing dialogues. You have to read the dialogue between two people and summarize it in no more than one sentence. The summary should be as short as possible, not re-explaining the dialogue in detail and using the person's name when implicitly mentioned."
    }
    FEW_SHOT_EXAMPLES = [
        {
            "context": "#Person1#: John, shall we go to Sun Store? I have decided to buy that Murrberry handbag. Anyway,I'm not carrying this one to Mary's wedding.\n#Person2#: But, Jane, why not rent one with Handbag Hire? Instead of $ 990,pay $ 50,and you have it for a whole week.\n#Person1#: Sounds great, but I never knew I can rent a handbag.\n#Person2#: Handbag Hire is a new business. It was founded two months ago. Its collection covers many designer handbags.\n#Person1#: So... for the price of one Murrberry, I can use a different bag each week for twenty weeks?\n#Person2#: Absolutely. And if you like one of them, you can choose to buy it at a discounted rate. Of course the price varies by age and condition. For example, a $ 1500 Murrberry bag can sell for just $750.\n#Person1#: Great, but how do I rent? By telephone? Or in person?\n#Person2#: Either. And more conveniently, it accepts online orders.\n#Person1#: I'll do it on line now. I still have one more question. Mary's wedding is next Saturday. There are only five days left. Do I have enough time?\n#Person2#: Don't worry. It promises that customers receive their orders by post within two days. Three more days to go.\n#Person1#: Oh, I'd better order one right now.",
            "answers": "Jane wants to buy that Murrberry handbag to carry to Mary's wedding, but John suggests renting one with Handbag Hire and tells her about the service in detail. Jane is pleased to have a try."
        },
        {
            "context": "#Person1#: The summers are so great here! Not hot at all. I love the cooling breezes, the clear air, all the greenery.\n#Person2#: This really has been a wonderful holiday for us. Shall we take a walk around the pond or into those woods for a while?\n#Person1#: Let's do both! Are we in a rush or anything\n#Person2#: No, not really. I had thought we'd stay in Hamburg tonight, but we can't unless we rush it. Let's stay in Bremen instead. Tomorrow we can have lunch in Hamburg, then check into a hostel in Copenhagen and have dinner there.\n#Person1#: Sounds fine to me. Whatever, let's enjoy this pond first.\n#Person2#: Sure. We can walk around to that path that leads into the woods there. Hey, look! There are some wild ducks over there in the reeds.\n#Person1#: I see them! Wow! How do you know they're wild?\n#Person2#: I used to go hunting with my uncle, that's how.\n#Person1#: They're neat. Now Let's take that path into the woods and see what we can see. . .",
            "answers": "#Person1# and #Person2# are enjoying a pond. #Person1# and #Person2# had planned to stay in Hamburg tonight, but they decide to stay in Bremen since they are not in a rush."
        },
        {
            "context": "#Person1#: Well Rebecca, is there anything else you need to know for now?\n#Person2#: I don't think so, Mr. Parsons. I think you have covered all the main points for me.\n#Person1#: Okay well listen, here is my business card with my mobile number. If any other questions spring to mind don't hesitate to contact me. Of course you can also call Miss Childs too.\n#Person2#: Great. Rmm, when can I expect to hear from you?\n#Person1#: Well, we are finishing the shortlist interviews tomorrow, so we will certainly have a decision made by early next week. Miss Childs will call you to discuss more on Monday or Tuesday. How does that so\n#Person2#: That sounds perfect. Thank you very much for taking the time to speak to me Mr. Parsons.\n#Person1#: The pleasure's all mine, Rebecca.\n#Person2#: I hope to hear from you very soon.\n#Person1#: Absolutely. Thanks for coming Rebecca. Goodbye.",
            "answers": "Mr. Parsons gives Rebecca his business card after the interview and tells Rebecca the decision will be made by early next week and Miss Childs will contact Rebecca."
        },
        {
            "context": "#Person1#: Trina, will you marry me?\n#Person2#: Yes! Yes! And yes! Jared, of course I'll marry you!\n#Person1#: Oh, Babe, I can't wait to spend the rest of my life with you! I can't wait for all the adventures we're going to have, for all the fights and the laughter. I can't wait to grow old and wrinkly with you.\n#Person2#: Oh, Jared! I can't wait for our wedding! I hope you don't mind, but I'Ve already chosen a date! Six months from now in the summer! Melissa saw you buying the ring last month so I'Ve had plenty of time to start planning!\n#Person1#: She what?\n#Person2#: Oh don't worry, sweetie, I didn't know when you were going to propose. It was still a nice surprise! As I was saying, I'Ve got it all planned out. There's almost nothing left to do! I wrote up our guest list and we will have roughly four hundred guests attending.\n#Person1#: Four hundred?\n#Person2#: No need to sweat it. My parents agreed to pay for most of the wedding, which is going to be low-budget anyway. So roughly four hundred people, which means that the hall at Northwoods Heights will be our reception venue. I thought it would be nice if we had the wedding at your parents'church and my uncle of course would be officiating. We'll meet with him soon for some pre-wedding counseling. The music for the wedding ceremony was a no-brainer. My step-sister and her string quartet will take care of that. My cousin will be the official photographer. I thought it would also be nice if his daughter could sing a solo. Did you know that she's going to be a professional opera singer?\n#Person1#: Ah. . .\n#Person2#: And then of course the ladies at the church would love to be our caterers for the banquet and we'll get the Youth Group to serve us. I was thinking that your friend's band could be our entertainment for the night. though they might have to tone it down a bit. Or we could hire a DJ. Your sister's husband could get us a discount with that company that does the decor at weddings. what's their name again? I was thinking that we could have an island paradise-themed wedding and our theme color would be a soothing blue like Aquamarine. And there will be a huge seashell on the wall behind the podium where we'll make our toasts! What do you think of small packages of drink mixes for our wedding favors? Who else am I missing? Oh, your uncle could be our florist and his wife could make our wedding cake!\n#Person1#: Wow.\n#Person2#: See? It's going to be wonderful! Oh this wedding is going to be everything I ever dreamed of.\n#Person1#: If I survive the next six months.",
            "answers": "Trina accepts Jared's proposal. Then, Jared is astonished to know that Trina already knew from Melissa who saw him buying the ring that he was planning this. Trina has chosen a date and has made a list of four hundred guests and she tells Jared about her arrangements in an ecstasy. Jared finds it hard to get through."
        },
        {
            "context": "#Person1#: Isabelle, you know I'm not interested in fame. \n#Person2#: Well, you don't seem to be interested in getting a real job, either. \n#Person1#: You know I'm interested in teaching. I'm looking for jazz students. . . \n#Person2#: Yeah, and every high school student in town is banging on your door, right? \n#Person1#: I know they're out there. I'll find them. \n#Person2#: You're such a dreamer! You think that you can spread the word of jazz in an underpass? ",
            "answers": "Isabelle thinks #Person1# is a dreamer because #Person1# doesn't do real things."
        },
        {
            "context": "#Person1#: Hi, Tony. You look unhappy. What's wrong? #Person2#: Oh, Steven, I made a big mistake. #Person1#: What happened? #Person2#: I really wish I hadn't done it. #Person1#: What on earth are you talking about? #Person2#: I got caught cheating. I feel so ashamed. The teacher saw me and told me I failed. #Person1#: What were you thinking? #Person2#: You know my father. If I fail, he'll kill me. I have to do well. #Person1#: But what you should do is study hard. #Person2#: I know. . . I know. . . it's all my fault. I feel awful that I didn't study, and I cheated, and I got caught. #Person1#: So long as you learn from your mistakes. ",
            "answers": "Tony got caught cheating and feels ashamed. Steven tells him to learn from it and study hard."
        },
        {
            "context": "#Person1#: Would you like to go to the party tonight? #Person2#: Whose party? #Person1#: Ruojia's. Don't you know that? Ruojia has got married. #Person2#: What! Is she really? I can't believe it! #Person1#: Yes. Yesterday. #Person2#: Good gracious. That's incredible! I feel so happy for her! #Person1#: Yes, me too. #Person2#: But how do you know that? #Person1#: I saw the news from her twitter. And she sent an email about it. #Person2#: What? I didn't receive it! #Person1#: Maybe you should check your email. #Person2#: Oh yes, I find it. Tonight at her home. Will you bring something? #Person1#: Yes, a pair of wineglasses and a card to wish her happy marriage. #Person2#: I will buy a tea set.",
            "answers": "#Person1# tells #Person2# that Ruojia is married and will have a party tonight. #Person2#'s surprised to know that. They will bring their gifts to bless her."
        }
    ]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    def create_few_shot(number_few_shot: int, **args):
        shot = []
        for i in range(number_few_shot):
            shot.append(
            [
                f"Dialogue: {FEW_SHOT_EXAMPLES[i]['context']}",
                f"Summary: {FEW_SHOT_EXAMPLES[i]['answers']}"
            ]
            )
        return shot

    def create_request(context="", **args):
        return [f"Dialogue: {context}", "Summary:"]
    
    
    def tokenization(items, tokenizer):
        return tokenizer(items["prompt"], padding='longest')
    
    def caculate_bertscore(predictions, references):
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type= "microsoft/deberta-xlarge-mnli",
            lang="en",
        )
        return results
    
    def create_prompt(task: str = "summary_dialogue", few_shot: int = 3, **args):
        prompt = ""
        prompt = DATASET_CONTEXT.get(task, "") + "\n\n"

        request = '\n'.join(create_request(**args))

        if few_shot:
            shot = '\n\n'.join(['\n'.join(s) for s in create_few_shot(few_shot, **args)])
            prompt += f"{shot}\n\n{request}"
        else:
            prompt += request
        return prompt


    def create_prompt_column(task: str = "summary_dialogue", few_shot: int = 3, item : dict = {}, has_title: bool = False):
        item['prompt'] = create_prompt(
                task, few_shot,
                title = item['title'] if has_title else "",
                context = item['dialogue'],
        )
        return item

    logging.basicConfig(level=logging.INFO)
    logging.info('Start')
    
    logging.info(f"Device: {DEVICE}")
    logging.info(f'Loading tokenizer {model_id}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / Path(model_id), use_fast=True)
    tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token}) 
    tokenizer.padding_side = 'left'
    logging.info('Tokenizer loaded.')
    logging.info('Loading model...')

    if use_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(MODEL_DIR / Path(model_id), device_map = "auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR / Path(model_id), device_map = "auto", torch_dtype=torch.bfloat16)

    model.resize_token_embeddings(len(tokenizer))
    context_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else tokenizer.model_max_length
    model.eval()
    logging.info('Model loaded.')
    
    
    dataset = load_from_disk(DATASET_DIR / Path(dataset_name))
    dataset = dataset[split_name]
    
    
    torch.cuda.empty_cache()
    
    
    answers = dataset.select_columns(["summary"])
    dataset = dataset.map(lambda item: create_prompt_column(
        item=item,
    ), num_proc= os.cpu_count())
    dataset = dataset.map(lambda item : tokenization(item, tokenizer), batched=True, num_proc=os.cpu_count())
    dataset = dataset.filter(lambda item: len(item['input_ids']) <= context_length)
    lengths = [len(item['input_ids']) for item in dataset]
    logging.info(f"Sequence length stats: Min={min(lengths)}, Max={max(lengths)}, Avg={sum(lengths)/len(lengths)}")
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids, attention_mask):
            padding_len = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.ones(padding_len, dtype=torch.long) * tokenizer.pad_token_id]))
            padded_attention_mask.append(torch.cat([mask, torch.zeros(padding_len, dtype=torch.long)]))
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask)
        }
    
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    logging.info(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count(), collate_fn=collate_fn)
    logging.info('Dataset processed...')
    
    torch.cuda.empty_cache()
    

    logging.info('Starting predictions...')
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            output = model.generate(
                batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                max_new_tokens=150,
                do_sample=False,
                eos_token_id= tokenizer.eos_token_id
            )
            output = output[:, len(batch['input_ids'][0]):]
            sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            for i in range(len(sentences)):
                sentences[i] = sentences[i].split('\n')[0].strip()
            predictions.append(sentences)

    predictions = list(chain(*predictions))
    answers = answers[:len(predictions)]

    output_file = EXPERIMENTS_RESULT_DIR / "predictions_and_answers.json"
    current_time = datetime.datetime.now().isoformat()  # Get current timestamp

    metadata = {
        "timestamp": current_time,
        "model_name": model_id,
        "batch_size": batch_size,
        "split_name": split_name,
        "gpu": "A100-80GB",
        "cpu": 16.0,
        "memory": 32768, 
    }

    with open(output_file, "w") as f:
        json.dump({"predictions": predictions, "answers": answers, "metadata": metadata}, f, indent=4)

    logging.info(f"Predictions and answers saved to {output_file}")
    logging.info(f"Metadata: {metadata}")
    experiments_result_volume.commit()
    
    results = caculate_bertscore(predictions, answers['summary'])
    average_f1 = sum(results['f1']) / len(results['f1'])
    logging.info(f"BertScore (average F1): {average_f1}")
    logging.info(f"BertScore (average Precision): {sum(results['precision']) / len(results['precision'])}")
    logging.info(f"BertScore (average Recall): {sum(results['recall']) / len(results['recall'])}")
    rouge_results = rouge_metric.compute(predictions=predictions, references=answers['summary'], use_stemmer=True, rouge_types=["rouge1", "rouge2", "rougeL"])
    logging.info(f"Rouge Result: {rouge_results}")

    logging.info('Predictions finished')
    logging.info('Computing scores...')
    
    
@app.local_entrypoint()
def main(*arglist):
    parser = argparse.ArgumentParser(description="Script to benchmark a model on a dataset.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    parser.add_argument("--model_tokenizer", type=str, help="Model tokenizer (default: model_id)")
    parser.add_argument("--dataset_id", type=str, help="Dataset hugging face ID")
    parser.add_argument("--split_name", type=str, default="test", help="Dataset split name")
    parser.add_argument("--context", action="store_true", help="To pre prompt an explanation of the task")
    parser.add_argument("--title", action="store_true", help="To keep title in the prompt")
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--bfloat", action="store_true", help="Load model in bf16")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions in txt file")
    parser.add_argument("--from_disk", action="store_true", help="Load dataset from disk")
    parser.add_argument("--task", type=str, default="summarization", help="Benchmark type (qa, qa_generative, summarization)")
    parser.add_argument("--mapping", type=str, default="", help="JSON file to map dataset column name")
    parser.add_argument("--mapping_dict", type=str, default="text", help="Field name in the answer dictionary.")
    parser.add_argument("--bert_score", action="store_true", help="To compute bert score")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    parser.add_argument("--context_length", type=int, default=None, help="Delete dataset row with length > context_length")
    parser.add_argument("--seq2seq", action="store_true", help="For encoder-decoder model")
    args = parser.parse_args(args = arglist)
    print(args)
    evaluate_model.remote(model_id = "weights_Llama-2-7b-hf_20250731_155656", use_peft=True)

