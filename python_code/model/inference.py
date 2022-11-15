import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


from python_code.model.inference_funcs import calc_sequence_likelihood
from python_code.model.model_utils import normalize, probablize

 
log_path = 'results/model/log'

# --- Hyperparameters --- #
schedule_steps = 5
batch_size_schedule = [50, 50, 50, 500, 5000]
batches_each_schedule_step = [5000000, 5000, 5000, 1000, 1000]
learning_rate_schedule = [1e-4, 1e-5, 1e-6, 1e-6, 1e-6]
momentum = 0.9

hyperparams_lines = f'batch_size_schedule = {batch_size_schedule} \n' + \
                    f'batches_each_schedule_step = {batches_each_schedule_step} \n' + \
                    f'learning_rate_schedule = {learning_rate_schedule} \n' + \
                    f'momentum = {momentum} \n' 


def probability_domain_penalty(probs):
    below_zero_penalty = F.relu(0 - probs)
    below_zero_penalty = below_zero_penalty.mean()
    below_zero_penalty = below_zero_penalty ** 2

    over_one_penalty = F.relu(probs - 1)
    over_one_penalty = over_one_penalty.mean()
    over_one_penalty = over_one_penalty ** 2

    penalty = below_zero_penalty + over_one_penalty
    return penalty 

def probability_sum_penalty(probs):
    penalty = probs.sum() - 1
    penalty = penalty / len(probs)
    penalty = penalty ** 2
    return penalty

def inference(model, data, ancestor_column='ancestor_alignment', descendant_column='sequence_alinment', only_synonymous=False, log_postfix=''):
    now = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
    log_dir = os.path.join(log_path, now + log_postfix)
    os.makedirs(log_dir)
    log_csv = os.path.join(log_dir, 'log.csv')

    with open(os.path.join(log_dir, 'hyperparams.txt'), 'w') as f:
        f.write(hyperparams_lines)

    print('-------------------------- Training start! ---------------------------')
    for name, param in model.named_parameters():
            ignore_names = 'motifs_prob'
            if not name.count(ignore_names):  # Not too long to print
                print(f'{name}: {param}')  
    print('----------------------------------------------------------------------')

    for step_counter in range(schedule_steps):

        batch_size = batch_size_schedule[step_counter]
        learning_rate = learning_rate_schedule[step_counter]
        batches_this_step = batches_each_schedule_step[step_counter]
        batch_counter = 0
        sample_counter = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss = torch.tensor([0.], requires_grad=True)
        mutations_counter = 0

        while batch_counter < batches_this_step:
            for _, row in data.sample(frac=1, ignore_index=True).iterrows():
                sample_counter += 1

                # Compute likelihood and accumulate loss
                targeting_probs, replication_probs = model(row[ancestor_column])
                negative_log_likelihood, n_mutations = calc_sequence_likelihood(row[ancestor_column], 
                                                                                row[descendant_column],
                                                                                targeting_probs, 
                                                                                replication_probs,
                                                                                only_synonymous=only_synonymous)

                loss = loss + negative_log_likelihood
                mutations_counter += n_mutations
                
                if sample_counter == batch_size:
                    sample_counter = 0

                    # Scale loss magnitude by the number of mutations
                    loss = loss / mutations_counter
                    mutations_counter = 0

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Prints
                    print(f'Learning step: {step_counter}, Batch: {batch_counter} / {batches_this_step}, Loss: {loss}')
                    
                    # Clipping to parameter space 
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                param.copy_(probablize(param))

                    print('----------------------------------------------------------------------')
                    for name, param in model.named_parameters():
                        ignore_names = 'motifs_prob'
                        if not name.count(ignore_names):  # Not too long to print
                            print(f'{name}: {param}')

                    print('----------------------------------------------------------------------')

                    # Log
                    log_line = {'loss': [loss.data.detach().item()]}

                    for name, param in model.named_parameters():
                        if name.count('motifs_prob'):  # Too long for csv lines
                            continue  

                        value = param.detach().numpy()
                        if len(param) > 1:
                            value = [value]
                        log_line[name] = value

                    pd.DataFrame(log_line).to_csv(log_csv, index=False, mode='a', header=not os.path.isfile(log_csv))

                    # Save parameters
                    if not (batch_counter % 100):
                        count = int(np.sum(batches_each_schedule_step[:step_counter]) + batch_counter)
                        save_parame_file = os.path.join(log_dir, 'state_dict_' + str(count))
                        torch.save(model.state_dict(), save_parame_file)

                    # Promote batch counter
                    batch_counter += 1
                    
                    # Reset loss
                    loss = torch.tensor([0.], requires_grad=True)

                    # Exit loop condition
                    if batch_counter == batches_this_step:
                        break





