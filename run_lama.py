from collections import defaultdict
from data_utils import load_dataset
from utils import *

def main():
    all_lamas = [1001,101,103,106,108,127,1303,131,136,1376,138,140,1412,159,17,176,178,19,
                 190,20,264,27,276,279,30,31,36,361,364,37,39,407,413,449,463,47,495,527,530,740,937]

    default_params = {
        'model': 'gpt2-xl',
        'dataset': None,
        'seed': None,
        'num_shots': None,
        'expr_name': None,
        'conditioned_on_correct_classes': False,
        'subsample_test_set': 300, # max cap the max number of samples from one template
        'unlabeled_pool_size': 300, # will not be used to compute cf
        'api_num_log_prob': 500,
    }

    # generate all params to try
    all_shots = [0, 1, 4, 8]
    num_seeds = 5

    all_params = []
    for which_lama in all_lamas:
        for num_shots in all_shots:
            for seed in range(num_seeds):
                p = deepcopy(default_params)
                p['dataset'] = f"lama_{which_lama}"
                p['seed'] = seed
                p['num_shots'] = num_shots
                p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                all_params.append(p)

    freeze_test_set = True; freeze_training_pool = True

    # experiment with each params
    all_results = []
    orig_accuracy_list = []
    calibrated_accuracy_list = []

    for param_index, params in enumerate(all_params):
        print(f"\n{params['expr_name']}")

        # load the data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)
        if params['template'] == "INVALID":
            orig_accuracy_list.append(-1)
            calibrated_accuracy_list.append(-1)
            continue

        # sample test set
        if params['subsample_test_set'] is None:
            test_sentences, test_labels = all_test_sentences, all_test_labels
            print(f"selecting full test set ({len(all_test_labels)})")
        else:
            if freeze_test_set:
                np.random.seed(0) # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
            sample_test_size = min(len(all_test_labels), params['subsample_test_set'])
            test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels, sample_test_size)
            print(f"selecting {len(test_labels)} subsample of test set")

        # sample unlabeled training pool
        if freeze_training_pool:
            np.random.seed(0)  # always use seed 0 result if freeze
        else:
            np.random.seed(params['seed'])
        sample_pool_size = min(len(all_train_labels), params['unlabeled_pool_size'])
        train_sentences_pool, train_labels_pool = random_sampling(all_train_sentences, all_train_labels, sample_pool_size)

        # sample training examples
        np.random.seed(params['seed'])
        train_sentences, train_labels = random_sampling(train_sentences_pool, train_labels_pool, params['num_shots'])

        # get all model responses
        all_responses, all_prompts = get_model_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=True)

        ### calculate calibrated accuracy
        # collect all possible options
        all_options = set()
        for resp in all_responses:
            logprobs = resp['logprobs']['top_logprobs'][0] # first token
            options = list(logprobs.keys())
            options = options[:min(100, len(options))]
            all_options.update(options)

        # get log prob for each option in the set
        cf_tokens = ["[MASK]", "N/A", "BLANK"]
        cf_probs_dict = defaultdict(lambda: [])

        if "gpt2" in params['model']:
            cf_prompts = []
            for entity in cf_tokens:
                prompt = params['prompt_func'](params, train_sentences, train_labels, entity, test_label_option=None)
                cf_prompts.append(prompt)

            all_resp = complete(cf_prompts, 1, model=params['model'], num_log_probs=50000)
            for resp in all_resp['choices']:
                log_prob = resp['logprobs']['top_logprobs'][0]
                for token, lp in log_prob.items():
                    cf_probs_dict[token].append(np.exp(lp))

        else:
            cf_prompts = []
            for option in all_options:
                for entity in cf_tokens:
                    prompt = params['prompt_func'](params, train_sentences, train_labels, entity, test_label_option=option)
                    cf_prompts.append(prompt)
            cf_prompts_chunked = list(chunks(cf_prompts, chunk_size_helper(params)))
            for chunk_id, prompt_chunk in enumerate(cf_prompts_chunked):
                all_resp = complete(prompt_chunk, 0, model=params['model'], echo=True, num_log_probs=1)
                for resp in all_resp['choices']:
                    log_prob = resp['logprobs']['token_logprobs'][-1]
                    token = resp['logprobs']['tokens'][-1]
                    prob = np.exp(log_prob)
                    cf_probs_dict[token].append(prob)

        new_cf_dict = {}
        for k, v in cf_probs_dict.items():
            new_cf_dict[k] = np.min(v) # Notice: Min across ensemble of placeholders
        cf_probs_dict = new_cf_dict

        all_calibrated_ans = []
        all_orig_ans = []
        error_count = 0
        total_count = 0
        for resp in all_responses:
            # get all probs
            orig_probs_list = []
            cf_probs_list = []
            all_tokens = []
            logprobs = resp['logprobs']['top_logprobs'][0]  # first token
            for token in list(logprobs.keys()):
                total_count += 1
                orig_prob = np.exp(logprobs[token])
                if token in cf_probs_dict.keys():
                    cf_prob = cf_probs_dict[token]
                    orig_probs_list.append(orig_prob)
                    cf_probs_list.append(cf_prob)
                    all_tokens.append(token)
                else: # hmm cannot find it
                    error_count += 1

            orig_probs_list = np.array(orig_probs_list)
            cf_probs_list = np.array(cf_probs_list)

            # normalize both original probs and cf probs so that both sum to 1
            orig_probs_list = orig_probs_list / np.sum(orig_probs_list)
            cf_probs_list = cf_probs_list / np.sum(cf_probs_list)


            # contextual calibration
            W = np.identity(len(orig_probs_list))
            b = -1 * np.expand_dims(cf_probs_list, axis=-1)
            calibrate_label_probs = np.matmul(W, np.expand_dims(orig_probs_list, axis=-1)) + b

            best_idx = np.argmax(calibrate_label_probs)
            best_idx_original = np.argmax(orig_probs_list)
            
            all_calibrated_ans.append(all_tokens[best_idx])
            all_orig_ans.append(all_tokens[best_idx_original])
            

        error_frac = error_count/total_count
        if error_frac > 0.01: print(f"WARNING: re-encode error frac: {error_frac:.2f}")

        orig_correctness_list = []
        orig_ans_list = []
        for model_ans, ans in zip(all_orig_ans, test_labels):
            model_ans = model_ans.strip()
            orig_ans_list.append(model_ans)
            if model_ans == ans:
                orig_correctness_list.append(1)
            else:
                orig_correctness_list.append(0)
        orig_correctness = np.mean(orig_correctness_list)
        print(f"Accuracy: {orig_correctness:.5f}")

        calibrated_correctness_list = []
        calibrated_ans_list = []
        for model_ans, ans in zip(all_calibrated_ans, test_labels):
            model_ans = model_ans.strip()
            calibrated_ans_list.append(model_ans)
            if model_ans == ans:
                calibrated_correctness_list.append(1)
            else:
                calibrated_correctness_list.append(0)
        calibrated_correctness = np.mean(calibrated_correctness_list)
        print(f"New accuracy: {calibrated_correctness:.5f}")

        
        

        orig_accuracy_list.append(orig_correctness)
        calibrated_accuracy_list.append(calibrated_correctness)

        ### savings
        result = dict()
        result['seed'] = params['seed']
        result['train_sentences'] = train_sentences
        result['train_labels'] = train_labels
        result['test_sentences'] = test_sentences
        result['test_labels'] = test_labels

        result['all_responses'] = all_responses
        result['cf_probs_dict'] = cf_probs_dict

        # answers
        result['orig_ans_list'] = orig_ans_list
        result['calibrated_ans_list'] = calibrated_ans_list

        # accuracies
        result['orig_correctness'] = orig_correctness
        result['calibrated_correctness'] = calibrated_correctness
        all_results.append(result)


    for p in all_params:
        p["single_prompt_func"] = None
        p["prompt_func"] = None
    all_results.insert(0, all_params)

    orig_accuracy_list = [acc for acc in orig_accuracy_list if acc >= 0]
    calibrated_accuracy_list = [acc for acc in calibrated_accuracy_list if acc >= 0]
    assert len(orig_accuracy_list) == len(calibrated_accuracy_list)


    orig_accuracy_list = np.reshape(orig_accuracy_list, (len(all_lamas), num_seeds))
    calibrated_accuracy_list = np.reshape(calibrated_accuracy_list, (len(all_lamas), num_seeds))

    combined_accuracy = np.mean(orig_accuracy_list, axis=0) # across 41 tasks
    calibrated_combined_accuracy = np.mean(calibrated_accuracy_list, axis=0) # across 41 tasks

    print(f"Original   | Mean: {np.mean(combined_accuracy):.4f}, Low: {np.min(combined_accuracy):.4f}, High: {np.max(combined_accuracy):.4f}, Std: {np.std(combined_accuracy):.4f}")
    print(f"Normalized | Mean: {np.mean(calibrated_combined_accuracy):.4f}, Low: {np.min(calibrated_combined_accuracy):.4f}, High: {np.max(calibrated_combined_accuracy):.4f}, Std: {np.std(calibrated_combined_accuracy):.4f}")

    # saving
    file_name = f"LAMA_{default_params['model']}_{all_shots[0]}shot_{repr(default_params['subsample_test_set'])}subsample_"

    from datetime import datetime
    dt_string = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    file_name += dt_string
    with open(file_name, 'wb') as f:
        pickle.dump(all_results, f)
    print("Saved to", file_name)


if __name__ == '__main__':
    main()
