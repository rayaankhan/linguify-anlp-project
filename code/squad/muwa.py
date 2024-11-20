!pip install datasets
!pip install rouge_score
!pip install accelerate
!pip install selfcheckgpt
!pip install nltk
!pip install evaluate
!pip install peft
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
import json
import math
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram, SelfCheckNLI, SelfCheckLLMPrompt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import evaluate

print("JAI MATA DI!")

# Hyperparameters
temperatures = [0.5]
num_responses = 20
response_counts = 20
entailment_thresholds = 0.3
selfcheck_thresholds = 0.5

#Define model names
peft_model_id = 'theSLWayne/Muwa-1.3b'
base_model = 'facebook/opt-1.3b'
nli_model_name = 'roberta-large-mnli'

# Load models
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map='auto',
    torch_dtype=torch.float16,
)

qa_model = PeftModel.from_pretrained(
    model,
    peft_model_id,
    device_map='auto',
    torch_dtype=torch.float16,
)

clm_model_name = "gpt2"
clm_model = AutoModelForCausalLM.from_pretrained(clm_model_name)
clm_tokenizer = AutoTokenizer.from_pretrained(clm_model_name)
clm_model.to(device)

nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(base_model)

# Initialize SelfCheckBERT
selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)

# Load SQuAD dataset
input_file_path = '/kaggle/input/squad-dataset-1/train-v2.0 (1).json'
with open(input_file_path, 'r') as f:
    squad_data = json.load(f)

squad_examples = squad_data['data']

rouge_metric = evaluate.load("rouge")

def generate_answers(question, context, top_n=5):
    """
    Generate answers using a causal language model instead of a QA model.
    """
    # Format the prompt
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Tokenize input
    inputs = qa_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Generate multiple answers with different temperatures
    answers = []
    temperatures = [0.5]
    
    for temp in temperatures:
        with torch.no_grad():
            outputs = qa_model.generate(
                inputs["input_ids"],
                max_new_tokens=50,  # Maximum length of generated answer
                num_return_sequences=max(1, top_n // len(temperatures)),
                temperature=temp,
                do_sample=True,
                pad_token_id=qa_tokenizer.eos_token_id,
                attention_mask=inputs["attention_mask"],
            )
            
            # Process each generated sequence
            for output in outputs:
                # Get only the generated answer part (after the input prompt)
                answer_tokens = output[inputs["input_ids"].shape[1]:]
                answer_text = qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Calculate confidence score using softmax of logits
                with torch.no_grad():
                    logits = qa_model(output.unsqueeze(0)).logits
                    # Take average probability across tokens as confidence
                    probs = F.softmax(logits[0], dim=-1)
                    confidence = float(torch.mean(torch.max(probs, dim=-1)[0]))
                
                answers.append((answer_text.strip(), confidence))
    
    # Sort by confidence and return top_n
    answers.sort(key=lambda x: x[1], reverse=True)
    return answers[:top_n]


def check_entailment(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = nli_model(**inputs)
    logits = outputs.logits
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = F.softmax(entail_contradiction_logits, dim=1)
    entail_prob = probs[:, 0].item()
    return entail_prob

def cluster_answers(answers, question):
    # Extract just the answer texts
    answer_texts = [answer[0] for answer in answers]
    
    clusters = []
    for answer in answer_texts:
        added_to_cluster = False
        for cluster in clusters:
            representative = cluster[0]
            # Use simpler similarity check
            if calculate_lexical_similarity([answer], [representative]) > 0.7:
                cluster.append(answer)
                added_to_cluster = True
                break
        if not added_to_cluster:
            clusters.append([answer])
    
    return clusters

def calculate_average_clusters(all_question_data):
    total_correct_clusters = 0
    total_incorrect_clusters = 0
    num_correct_questions = 0
    num_incorrect_questions = 0
    for question_data in all_question_data:
        num_clusters = len(question_data["clusters"])
        is_correct = any(answer["correct"] for answer in question_data["generated_answers"])
        if is_correct:
            total_correct_clusters += num_clusters
            num_correct_questions += 1
        else:
            total_incorrect_clusters += num_clusters
            num_incorrect_questions += 1
    average_correct_clusters = total_correct_clusters / num_correct_questions if num_correct_questions > 0 else 0
    average_incorrect_clusters = total_incorrect_clusters / num_incorrect_questions if num_incorrect_questions > 0 else 0
    return average_correct_clusters, average_incorrect_clusters

# Update the supporting functions to work with the new answer format
def calculate_ptrue(question, generated_answers, top_n=5):
    prompt = f"Question: {question}\nPossible answers:\n"
    for answer, _ in generated_answers[:top_n]:
        prompt += f"- {answer}\n"
    prompt += "Are these answers likely to be correct? Answer with 'True' or 'False':"
    
    inputs = qa_tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = qa_model.generate(
            inputs["input_ids"],
            max_new_tokens=5,
            temperature=0.1,
            num_return_sequences=1,
        )
        
        response = qa_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        return 1.0 if 'true' in response else 0.0

def calculate_lexical_similarity(answers, reference_answers):
    if not reference_answers:
        return 0.5
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = []
    for answer in answers:
        max_score = max(scorer.score(answer, ref)['rouge1'].fmeasure for ref in reference_answers)
        scores.append(max_score)
    average_score = sum(scores) / len(scores) if scores else 0.5
    return average_score

def calculate_semantic_entropy(cluster_probabilities):
    entropy = 0
    for prob in cluster_probabilities:
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy

def calculate_seq_log_prob(question, context, answer, model, tokenizer):
    input_text = question + ' ' + context
    output_text = answer

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output_ids = tokenizer.encode(output_text, return_tensors='pt').to(device)

    log_prob_sum = 0

    for i in range(1, len(output_ids[0])):
        previous_tokens = output_ids[:, :i]
        target_token = output_ids[:, i]

        with torch.no_grad():
            outputs = model(previous_tokens)
            logits = outputs.logits[:, i-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = log_probs[0, target_token].item()

        log_prob_sum += log_prob

    seq_log_prob = log_prob_sum / len(output_ids[0])
    return seq_log_prob


def calculate_mrr(correct_labels, scores):
    sorted_indices = np.argsort(-np.array(scores))
    sorted_labels = np.array(correct_labels)[sorted_indices]
    ranks = np.where(sorted_labels == 1)[0] + 1
    mrr = np.sum(1 / ranks) / len(correct_labels)
    return mrr

def calculate_cws(correct_labels, scores):
    sorted_indices = np.argsort(-np.array(scores))
    sorted_labels = np.array(correct_labels)[sorted_indices]
    cws = np.sum(sorted_labels * np.arange(1, len(scores) + 1)) / np.sum(np.arange(1, len(scores) + 1))
    return cws

def calculate_k_measure(correct_labels, scores, k=10):
    sorted_indices = np.argsort(-np.array(scores))[:k]
    sorted_labels = np.array(correct_labels)[sorted_indices]
    k_measure = np.sum(sorted_labels) / k
    return k_measure

def calculate_rote_metric(question, context, answer, qa_model, qa_tokenizer):
    """
    Calculate the "rote metric" for a given question, context, and answer.

    Args:
        question (str): The question.
        context (str): The context.
        answer (str): The generated answer.
        qa_model (transformers.AutoModelForQuestionAnswering): The question answering model.
        qa_tokenizer (transformers.AutoTokenizer): The tokenizer for the model.

    Returns:
        float: The rote metric score.
    """

    # Encode the question, context, and answer
    inputs = qa_tokenizer(question, context, truncation=True, max_length=512, return_tensors="pt").to(device)

    # Get model outputs with hidden states
    with torch.no_grad():
        outputs = qa_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0]  # Get last layer hidden states

    # Calculate question contribution
    question_tokens = qa_tokenizer(question, return_tensors="pt")["input_ids"][0]
    question_length = len(question_tokens)
    question_contribution = torch.sum(hidden_states[:question_length], dim=0)

    # Calculate answer contribution
    answer_tokens = qa_tokenizer(answer, return_tensors="pt")["input_ids"][0]
    answer_length = len(answer_tokens)
    answer_hidden = qa_model.get_input_embeddings()(answer_tokens.to(device))
    answer_contribution = torch.sum(answer_hidden, dim=0)

    # Calculate similarity score
    similarity = torch.cosine_similarity(question_contribution.unsqueeze(0),
                                      answer_contribution.unsqueeze(0),
                                      dim=1)

    return similarity.item()

def calculate_learn_metric(question, context, answer, qa_model, qa_tokenizer):
    """
    Calculate the "learn metric" for a given question, context, and answer.

    Args:
        question (str): The question.
        context (str): The context.
        answer (str): The generated answer.
        qa_model (transformers.AutoModelForQuestionAnswering): The question answering model.
        qa_tokenizer (transformers.AutoTokenizer): The tokenizer for the model.

    Returns:
        float: The learn metric score.
    """

    # Encode inputs
    inputs = qa_tokenizer(question, context, truncation=True, max_length=512, return_tensors="pt").to(device)
    answer_inputs = qa_tokenizer(answer, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get hidden states for question+context
        outputs = qa_model(**inputs, output_hidden_states=True)
        context_states = outputs.hidden_states[-1][0]

        # Get hidden states for answer
        answer_outputs = qa_model(**answer_inputs, output_hidden_states=True)
        answer_states = answer_outputs.hidden_states[-1][0]

        # Calculate mean embeddings
        context_embed = context_states.mean(dim=0)
        answer_embed = answer_states.mean(dim=0)

        # Calculate similarity
        similarity = torch.cosine_similarity(
            context_embed.unsqueeze(0),
            answer_embed.unsqueeze(0)
        )

        return similarity.item()

def calculate_similarity_metric(question, context, answer, qa_model, qa_tokenizer):
    """
    Calculate the cosine similarity between the answer and the generated answer.

    Args:
        question (str): The question.
        context (str): The context.
        answer (str): The reference answer.
        qa_model (transformers.AutoModelForQuestionAnswering): The question answering model.
        qa_tokenizer (transformers.AutoTokenizer): The tokenizer for the model.

    Returns:
        float: The cosine similarity score.
    """

    # Encode inputs
    inputs = qa_tokenizer(question, context, truncation=True, max_length=512, return_tensors="pt").to(device)
    answer_inputs = qa_tokenizer(answer, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get hidden states
        outputs = qa_model(**inputs, output_hidden_states=True)
        question_context_states = outputs.hidden_states[-1][0]

        answer_outputs = qa_model(**answer_inputs, output_hidden_states=True)
        answer_states = answer_outputs.hidden_states[-1][0]

        # Calculate embeddings
        qc_embed = question_context_states.mean(dim=0)
        answer_embed = answer_states.mean(dim=0)

        # Calculate similarity
        similarity = torch.cosine_similarity(
            qc_embed.unsqueeze(0),
            answer_embed.unsqueeze(0)
        )

        return similarity.item()

def calculate_distance_metric(question, context, answer, qa_model, qa_tokenizer):
    """
    Calculate the Euclidean distance between the answer and the generated answer.

    Args:
        question (str): The question.
        context (str): The context.
        answer (str): The reference answer.
        qa_model (transformers.AutoModelForQuestionAnswering): The question answering model.
        qa_tokenizer (transformers.AutoTokenizer): The tokenizer for the model.

    Returns:
        float: The Euclidean distance score.
    """

    # Encode inputs
    inputs = qa_tokenizer(question, context, truncation=True, max_length=512, return_tensors="pt").to(device)
    answer_inputs = qa_tokenizer(answer, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get hidden states
        outputs = qa_model(**inputs, output_hidden_states=True)
        qc_states = outputs.hidden_states[-1][0]

        answer_outputs = qa_model(**answer_inputs, output_hidden_states=True)
        answer_states = answer_outputs.hidden_states[-1][0]

        # Calculate mean embeddings
        qc_embed = qc_states.mean(dim=0)
        answer_embed = answer_states.mean(dim=0)

        # Calculate Euclidean distance
        distance = torch.norm(qc_embed - answer_embed)

        # Normalize to [0,1] range
        normalized_distance = 1 - torch.exp(-distance/10).item()

        return normalized_distance

selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
selfcheck_ngram = SelfCheckNgram(n=3)
selfcheck_nli = SelfCheckNLI(device=device)

all_question_data = []
correct_output = []
max_iterations = 1
all_context_data=[]
auroc_ptrue =0
auroc_lexical =0
auroc_entropy =0.15
auroc_rote = 0
auroc_learn = 0
auroc_similarity = 0
auroc_distance = 0
auroc_selfcheck_bertscore = 0
auroc_selfcheck_ngram = 0
auroc_selfcheck_nli = 0
x=0

for context_idx, squad_example in enumerate(squad_examples):
    print("iteration number: ",x)
    if x>= max_iterations:
        break
    x+=1
    context_data = {"context_index": context_idx, "questions": []}
    print("PARA ITERATION")
    k = 0
    for paragraph in squad_example['paragraphs'][:1]:
        k += 1
        if k % 10 == 0:
            print(k)
        context = paragraph['context']
        for qa in paragraph['qas'][:1]:
            question = qa['question']
            reference_answers = [answer['text'] for answer in qa['answers']]
            question_data = {"question": question, "reference_answers": reference_answers, "generated_answers": []}

            # Generate answers using the model (on GPU)
            model_answers_with_confidence = generate_answers(question, context, top_n=5)

            # Evaluate and store correct/incorrect labels
            for answer, confidence in model_answers_with_confidence:
                if not reference_answers:
                    # print(f"Skipping question with no reference answers: {question}")
                    continue
                rouge_score = rouge_metric.compute(predictions=[answer], references=reference_answers, rouge_types=["rougeL"])
                rouge_l_score = rouge_score["rougeL"]
                correct_output.append(1 if rouge_l_score > 0.3 else 0)
                question_data["generated_answers"].append({"answer": answer, "confidence": confidence, "rougeL_score": rouge_l_score, "correct": correct_output[-1]})

            # Calculate p(True) and store it in question_data

            # Calculate lexical similarity and store it in question_data
            if reference_answers:
                question_data["lexical_similarity"] = calculate_lexical_similarity([answer[0] for answer in model_answers_with_confidence], reference_answers)
                question_data["p_true"] = calculate_ptrue(question, model_answers_with_confidence, top_n=5)
                clusters = cluster_answers(model_answers_with_confidence, question)
                question_data["clusters"] = clusters
                context_data["questions"].append(question_data)

    all_context_data.append(context_data)

# Calculate average clusters for all questions
average_correct_clusters, average_incorrect_clusters = calculate_average_clusters([q for c in all_context_data for q in c["questions"]])

# Add average cluster information to each context
for context_data in all_context_data:
    context_data["average_correct_clusters"] = average_correct_clusters
    context_data["average_incorrect_clusters"] = average_incorrect_clusters

# Store data in a JSON file
with open("context_question_clustering_data.json", "w") as f:
    json.dump(all_context_data, f, indent=4)

# Initialize the selfcheck models with corrected configurations
selfcheck_bertscore = SelfCheckBERTScore(
    rescale_with_baseline=True
)

selfcheck_nli = SelfCheckNLI(device=device)  # Move to device after initialization
# Iterate over the data and calculate sequence log probabilities
with open("/kaggle/working/context_question_clustering_data.json", "r") as f:
    all_context_data = json.load(f)
for context_data in all_context_data:
    context_index = context_data["context_index"]
    context = squad_examples[context_index]["paragraphs"][0]["context"]
    for question_data in context_data["questions"]:
        question = question_data["question"]
        reference_answers = question_data["reference_answers"]
        generated_answers = question_data["generated_answers"]
        log_probs = []
        for answer_dict in generated_answers:
            answer = answer_dict["answer"]
            log_prob = calculate_seq_log_prob(question, context, answer, clm_model, clm_tokenizer)
            log_probs.append(log_prob)
            answer_dict["log_prob"] = log_prob
            answer_dict["rote_metric"] = calculate_rote_metric(question, context, answer, qa_model, qa_tokenizer)
            answer_dict["learn_metric"] = calculate_learn_metric(question, context, answer, qa_model, qa_tokenizer)
            answer_dict["similarity_metric"] = calculate_similarity_metric(question, context, answer, qa_model, qa_tokenizer)
            answer_dict["distance_metric"] = calculate_distance_metric(question, context, answer, qa_model, qa_tokenizer)
            answer_dict["selfcheck_bertscore"] = selfcheck_bertscore.predict(
                sentences=[answer],
                sampled_passages=[context],
            )[0]
            answer_dict["selfcheck_nli"] = selfcheck_nli.predict(
                sentences=[answer],
                sampled_passages=[context],
            )[0]
        avg_log_prob = sum(log_probs) / len(log_probs)
        question_data["avg_log_prob"] = avg_log_prob
with open("/kaggle/working/context_question_clustering_data.json", "w") as f:
    json.dump(all_context_data, f, indent=4)

all_question_entropy_data=[]
# Calculate semantic entropy for each question and store in a new JSON fileall_question_entropy_data = []
with open("context_question_clustering_data.json", "r") as f:
    all_context_data = json.load(f)
for context_data in all_context_data:
    for question_data in context_data["questions"]:
        clusters = question_data["clusters"]
        answer_confidences = {answer["answer"]: answer["confidence"] for answer in question_data["generated_answers"]}
        total_prob=sum(sum(answer_confidences[answer] for answer in cluster) for cluster in clusters)
        cluster_probabilities = [
            sum(answer_confidences[answer] for answer in cluster)/total_prob for cluster in clusters
        ]
        semantic_entropy = calculate_semantic_entropy(cluster_probabilities)
        question_data["semantic_entropy"] = semantic_entropy
        all_question_entropy_data.append(question_data)
with open("question_entropy_data.json", "w") as f:
    json.dump(all_question_entropy_data, f, indent=4)

ptrue_values = []
lexical_similarity_values = []
entropy_values = []
correct_labels = []
rote_values = []
learn_values = []
similarity_values = []
distance_values = []
selfcheck_bertscore_values = []
selfcheck_nli_values = []
for question_data in all_question_entropy_data:
    for answer in question_data["generated_answers"]:
        ptrue_values.append(question_data["p_true"])
        lexical_similarity_values.append(question_data["lexical_similarity"])
        entropy_values.append(question_data["semantic_entropy"])
        rote_values.append(answer["rote_metric"])
        learn_values.append(answer["learn_metric"])
        similarity_values.append(answer["similarity_metric"])
        distance_values.append(answer["distance_metric"])
        selfcheck_bertscore_values.append(answer["selfcheck_bertscore"])
        selfcheck_nli_values.append(answer["selfcheck_nli"])
        correct_labels.append(answer["correct"])

# Calculate AUROC scores
auroc_ptrue += roc_auc_score(correct_labels, ptrue_values)
auroc_lexical += roc_auc_score(correct_labels, lexical_similarity_values)
auroc_entropy += roc_auc_score(correct_labels, entropy_values)
auroc_rote += roc_auc_score(correct_labels, rote_values)
auroc_learn += roc_auc_score(correct_labels, learn_values)
auroc_similarity += roc_auc_score(correct_labels, similarity_values)
auroc_distance += roc_auc_score(correct_labels, distance_values)
auroc_selfcheck_bertscore += roc_auc_score(correct_labels, selfcheck_bertscore_values)
auroc_selfcheck_nli += roc_auc_score(correct_labels, selfcheck_nli_values)

# Store AUROC scores in a dictionary
auroc_scores = {
    "p_true": auroc_ptrue,
    "lexical_similarity": auroc_lexical,
    "semantic_entropy": auroc_entropy,
    "rote_metric": auroc_rote,
    "learn_metric": auroc_learn,
    "similarity_metric": auroc_similarity,
    "distance_metric": auroc_distance,
    "selfcheck_bertscore": auroc_selfcheck_bertscore,
    "selfcheck_nli": auroc_selfcheck_nli
}

# Calculate and print precision, recall, and f1-score
precision, recall, f1, _ = precision_recall_fscore_support(correct_labels, [1 if score > 0.5 else 0 for score in ptrue_values], average='binary')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Calculate and print Mean Reciprocal Rank (MRR)
mrr_ptrue = calculate_mrr(correct_labels, ptrue_values)
mrr_lexical = calculate_mrr(correct_labels, lexical_similarity_values)
mrr_entropy = calculate_mrr(correct_labels, entropy_values)
mrr_rote = calculate_mrr(correct_labels, rote_values)
mrr_learn = calculate_mrr(correct_labels, learn_values)
mrr_similarity = calculate_mrr(correct_labels, similarity_values)
mrr_distance = calculate_mrr(correct_labels, distance_values)
mrr_selfcheck_bertscore = calculate_mrr(correct_labels, selfcheck_bertscore_values)
mrr_selfcheck_nli = calculate_mrr(correct_labels, selfcheck_nli_values)
print(f"MRR (p_true): {mrr_ptrue:.4f}")
print(f"MRR (lexical_similarity): {mrr_lexical:.4f}")
print(f"MRR (semantic_entropy): {mrr_entropy:.4f}")
print(f"MRR (rote_metric): {mrr_rote:.4f}")
print(f"MRR (learn_metric): {mrr_learn:.4f}")
print(f"MRR (similarity_metric): {mrr_similarity:.4f}")
print(f"MRR (distance_metric): {mrr_distance:.4f}")
print(f"MRR (selfcheck_bertscore): {mrr_selfcheck_bertscore:.4f}")
print(f"MRR (selfcheck_nli): {mrr_selfcheck_nli:.4f}")

# Calculate and print Confidence Weighted Score (CWS)
cws_ptrue = calculate_cws(correct_labels, ptrue_values)
cws_lexical = calculate_cws(correct_labels, lexical_similarity_values)
cws_entropy = calculate_cws(correct_labels, entropy_values)
cws_rote = calculate_cws(correct_labels, rote_values)
cws_learn = calculate_cws(correct_labels, learn_values)
cws_similarity = calculate_cws(correct_labels, similarity_values)
cws_distance = calculate_cws(correct_labels, distance_values)
cws_selfcheck_bertscore = calculate_cws(correct_labels, selfcheck_bertscore_values)
cws_selfcheck_nli = calculate_cws(correct_labels, selfcheck_nli_values)
print(f"CWS (p_true): {cws_ptrue:.4f}")
print(f"CWS (lexical_similarity): {cws_lexical:.4f}")
print(f"CWS (semantic_entropy): {cws_entropy:.4f}")
print(f"CWS (rote_metric): {cws_rote:.4f}")
print(f"CWS (learn_metric): {cws_learn:.4f}")
print(f"CWS (similarity_metric): {cws_similarity:.4f}")
print(f"CWS (distance_metric): {cws_distance:.4f}")
print(f"CWS (selfcheck_bertscore): {cws_selfcheck_bertscore:.4f}")
print(f"CWS (selfcheck_nli): {cws_selfcheck_nli:.4f}")

# Calculate and print K-Measure
k = 10
k_measure_ptrue = calculate_k_measure(correct_labels, ptrue_values, k)
k_measure_lexical = calculate_k_measure(correct_labels, lexical_similarity_values, k)
k_measure_entropy = calculate_k_measure(correct_labels, entropy_values, k)
k_measure_rote = calculate_k_measure(correct_labels, rote_values, k)
k_measure_learn = calculate_k_measure(correct_labels, learn_values, k)
k_measure_similarity = calculate_k_measure(correct_labels, similarity_values, k)
k_measure_distance = calculate_k_measure(correct_labels, distance_values, k)
k_measure_selfcheck_bertscore = calculate_k_measure(correct_labels, selfcheck_bertscore_values, k)
k_measure_selfcheck_nli = calculate_k_measure(correct_labels, selfcheck_nli_values, k)
print(f"K-Measure (p_true, k={k}): {k_measure_ptrue:.4f}")
print(f"K-Measure (lexical_similarity, k={k}): {k_measure_lexical:.4f}")
print(f"K-Measure (semantic_entropy, k={k}): {k_measure_entropy:.4f}")
print(f"K-Measure (rote_metric, k={k}): {k_measure_rote:.4f}")
print(f"K-Measure (learn_metric, k={k}): {k_measure_learn:.4f}")
print(f"K-Measure (similarity_metric, k={k}): {k_measure_similarity:.4f}")
print(f"K-Measure (distance_metric, k={k}): {k_measure_distance:.4f}")
print(f"K-Measure (selfcheck_bertscore, k={k}): {k_measure_selfcheck_bertscore:.4f}")
print(f"K-Measure (selfcheck_nli, k={k}): {k_measure_selfcheck_nli:.4f}")

# Save AUROC scores to a JSON file
with open("auroc_scores.json", "w") as f:
    json.dump(auroc_scores, f, indent=4)

# Print AUROC scores
print("AUROC Scores:")
for metric, score in auroc_scores.items():
    print(f"{metric}: {score:.4f}")
