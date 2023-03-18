import json
import numpy as np
from tqdm import tqdm
import mauve
from transformers import AutoTokenizer 
from simctg.evaluation import measure_repetition_and_diversity
from simcse import SimCSE
from Bleu import Bleu
from SelfBleu import SelfBleu

import os
# Set TOKENIZERS_PARALLELISM to 'true' or 'false' to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"


###############
# Data Loader #
###############
def load_gpt2_dataset(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    for i, line in tqdm(enumerate(open(json_file_name))):
        try:
            texts.append(json.loads(line)["text"])
        except json.decoder.JSONDecodeError:
            continue
   
    return texts # [gen_text1, gen_text2, ...]


def load_get2_pair(json_file_name, num_examples=float("inf")):
    texts = []
    print(f"Loading data from {json_file_name}......")
    for i, line in tqdm(enumerate(open(json_file_name))):
        try:
            texts.append((json.loads(line)["prompt_text"], json.loads(line)["gen_text"]))
        except json.decoder.JSONDecodeError:
            continue

    return texts # [(prompt_text1, gen_text1), (prompt_text2, gen_text2), ...]


#########
# MAUVE #
#########
def compute_mauve(human_text, gen_text, max_len):
    """
    Compute the MAUVE score of given text with reference to webtext.
    
    :param human_text: human text (webtext)
    :param gen_text: model-generated text
    :param max_len: maximum text length to truncate
    :return mauve_score: MAUVE score of given text with reference to webtext
    
    Results: webtext_train(9310) vs. opt_13b-125m(5273) --> mauve = 0.8627802521742031 (1024) / 0.9102370353331146 (256) / 0.9266521248714534 (128)
             webtext_valid(5000) vs. opt_13b-125m(5273) --> mauve = 0.8735936367558702 (1024) / 0.9218938670439518 (256) / 0.9272530776764185 (128)
             webtext_test(5000)  vs. opt_13b-125m(5273) --> mauve = 0.8552726757055065 (1024) / 0.9187265446563135 (256) / 0.9203297293141568 (128)
    Finding: The lower tgt_len, the better performance.
    """
    
    # call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to {tgt_len} tokens
    mauve_score = mauve.compute_mauve(p_text=human_text, q_text=gen_text, device_id=0, max_text_length=max_len, verbose=False, featurize_model_name="gpt2").mauve
    
    return mauve_score


##########################
# Repetition & Diversity #
##########################
def compute_rep_div(gen_text):
    """
    Compute the rep-2, rep-3, rep-4 and diversity scores of given text.
    
    :param gen_text: model-generated text
    :return rep_2: 2-gram repetition score of given text
    :return rep_3: 3-gram repetition score of given text
    :return rep_4: 4-gram repetition score of given text
    :return div_score: diversity score of given text
    
    Results: rep-2 score: 8.25 [with batch_decode]
             rep-3 score: 4.17
             rep-4 score: 2.95
             diversity score: 0.8533026626250001
    Finding: The lower tgt_len, the better performance. batch_decode is involved as it does help produce better scores.
    """
    
    rep_2, rep_3, rep_4, div_score = measure_repetition_and_diversity(gen_text)
    
    return rep_2, rep_3, rep_4, div_score


#############
# Coherence #
#############
def compute_coh(file_name):
    """
    Compute the coherence score of given text with reference to its prefix.
    
    :param file_name: jsonl file which stores <prompt_text, gen_text> pairs
    :return coh_score: coherence score of given text with reference to its prefix
    
    Results: coherence score: 0.8022059978309312 [w/o batch_decode]
    Finding: Indepedent of {tgt_len}.
    """
    
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    sent_lst = load_get2_pair(file_name) # "opttext_pair.jsonl"
    pp_lst, yy_lst = zip(*sent_lst)
    similarities = np.array(model.similarity(list(pp_lst), list(yy_lst)))
    coh_score = similarities.trace() / len(similarities) 
    
    return coh_score


########
# BLEU #
########
def compute_bleu(human_text, gen_text):
    """
    Compute the BLEU score of given text with reference to webtext.
    
    :param human_text: human text (webtext)
    :param gen_text: model-generated text
    :return bleu_score: BLEU score of given text with reference to webtext
    
    Results: webtext_train(9310) vs. opt_13b-125m(5273) --> bleu = 0.3199899896719573 (1024) / 0.2445160862568812 (256) / 0.20247434364644207 (128) [with batch_decode]
             webtext_valid(5000) vs. opt_13b-125m(5273) --> bleu = 0.3174006504263524 (1024) / 0.24456896382229462 (256) / 0.2013412137537578 (128)
             webtext_test(5000)  vs. opt_13b-125m(5273) --> bleu = 0.3174536690014808 (1024) / 0.24404910401572574 (256) / 0.20054597183508927 (128)
    Finding: The higher tgt_len, the better performance.
    """
    
    bleu = Bleu()
    bleu.real_data = human_text # human text
    bleu.test_data = gen_text # model text
    bleu_score = bleu.get_score()
    
    return bleu_score


#############
# Self-BLEU #
#############
def compute_self_bleu(gen_text):
    """
    Compute the Self-BLEU score of given text.
    
    :param gen_text: model-generated text
    :return self_bleu_score: Self-BLEU score of given text
    
    Results: 0.3786205616072991 (1024) / 0.372401794537256 (256) / 0.3283484372757455 (128) [with batch_decode]
    Finding: The higher tgt_len, the better performance.
    """
    
    self_bleu = SelfBleu()
    self_bleu.test_data = gen_text # model text
    self_bleu_score = self_bleu.get_score()
    
    return self_bleu_score


if __name__ == "__main__":
    # hyper-parameters
    tgt_len = 128   # max text length (1024 / 256 / 128); 128 is used in Contrastive Decoding code
    split = "train" # reference data source (train / valid / test)
  
    # load original human & model texts
    p_text_ = load_gpt2_dataset(f"data/webtext.{split}.jsonl") # human text
    q_text_ = load_gpt2_dataset("data/webtext.train_opt.jsonl") # model text

    # tokenization & batch_decode
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    x = tokenizer(p_text_, truncation=True, max_length=tgt_len)["input_ids"]
    y = tokenizer(q_text_, truncation=True, max_length=tgt_len)["input_ids"]
    print("Performing batch_decode......")
    xxyy = [(xx, yy) for (xx, yy) in tqdm(zip(x, y), total=min(len(x), len(y))) if len(xx) <= tgt_len and len(yy) <= tgt_len]
    x, y = zip(*xxyy)

    # map back to texts
    p_text = tokenizer.batch_decode(x) # [:target_num]
    q_text = tokenizer.batch_decode(y) # [:target_num]

    # compute scores
    mauve_score = compute_mauve(p_text, q_text, tgt_len)
    print("mauve score:", mauve_score)

    rep_2, rep_3, rep_4, div_score = compute_rep_div(q_text)
    print("rep-2 score:", rep_2)
    print("rep-3 score:", rep_3)
    print("rep-4 score:", rep_4)
    print("diversity score:", div_score)

    coh_score = compute_coh(file_name="opttext_pair.jsonl")
    print("coherence score:", coh_score)

    bleu_score = compute_bleu(p_text, q_text)
    print("bleu score:", bleu_score)

    self_bleu_score = compute_self_bleu(q_text)
    print("self-bleu score:", self_bleu_score)
            