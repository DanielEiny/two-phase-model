import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from python_code.data_preprocess.count_mutations import mismatch_positions

# --- Hyperparameters --- #
eps = 1e-7
schedule_steps = 5
batch_size_schedule = [100, 1000, 1000, 10000, 100000]
batches_each_schedule_step = [10000, 5000, 1000, 1000, 1000]
learning_rate_schedule = [1e-3, 1e-3, 1e-3, 1e-3, 1e-4]
r1 = 100  # probability domain panelty coefficient
r2 = 10  # probability sum panelty coefficient
log_path = 'results/model/log'

def calc_target_likelihood(ancestor_nucleotide, descendant_nucleotide, targeting_prob, replication_prob):
    likelihood = torch.tensor([0.], requires_grad=True) 
    
    each_nucleotid_substitution_prob = 1 / 3
    likelihood = likelihood + targeting_prob * (1 - replication_prob) * each_nucleotid_substitution_prob
    
    # Cases of possibly replication
    if (ancestor_nucleotide == 'C' and descendant_nucleotide == 'T') or \
       (ancestor_nucleotide == 'G' and descendant_nucleotide == 'A'):  
        likelihood = likelihood + targeting_prob * replication_prob

    return -torch.log(likelihood + eps)

def calc_sequence_likelihood(ancestor_sequence, descendant_sequence, targeting_probs, replication_probs):
    likelihood = torch.tensor([0.], requires_grad=True) 
    targets = mismatch_positions(ancestor_sequence, descendant_sequence)
        
    for t in targets:
        likelihood = likelihood + calc_target_likelihood(ancestor_sequence[t], 
                                                         descendant_sequence[t],
                                                         targeting_probs[t],
                                                         replication_probs[t])
        # Compensate no-replacement effect
        targeting_probs = targeting_probs.index_fill(0, torch.LongTensor([t]), 0)
        targeting_probs = targeting_probs / targeting_probs.sum()

    return likelihood, len(targets)

def probability_domain_penalty(prob):
    below_zero_penalty = (F.relu(0 - prob)).sum() ** 2
    over_one_penalty = (F.relu(prob - 1)).sum() ** 2
    return below_zero_penalty + over_one_penalty

def probability_sum_penalty(probs):
    return (probs.sum() - 1) ** 2

def inference(model, data, ancestor_column='ancestor_alignment', descendant_column='sequence_alinment'):
    now = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
    log_dir = os.path.join(log_path, now)
    os.makedirs(log_dir)
    log_csv = os.path.join(log_dir, 'log.csv')

    print('-------------------------- Training start! ---------------------------')
    for name, param in model.named_parameters():
        print(f'{name}: {param}')
    print('----------------------------------------------------------------------')

    for step_counter in range(schedule_steps):

        batch_size = batch_size_schedule[step_counter]
        learning_rate = learning_rate_schedule[step_counter]
        batches_this_step = batches_each_schedule_step[step_counter]
        batch_counter = 0
        sample_counter = 0

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss = torch.tensor([0.], requires_grad=True)
        mutations_counter = 0

        while batch_counter < batches_this_step:
            for _, row in data.sample(frac=1, ignore_index=True).iterrows():
                sample_counter += 1

                # Compute likelihood and accumulate loss
                targeting_probs, replication_probs = model(row[ancestor_column])
                negative_log_likelihood, n_mutations = calc_sequence_likelihood(row[ancestor_column], row[descendant_column], targeting_probs, replication_probs)
                loss = loss + negative_log_likelihood
                mutations_counter += n_mutations
                
                if sample_counter == batch_size:
                    sample_counter = 0
                    batch_counter += 1

                    # Scale loss magnitude by the number of mutations
                    loss = loss / mutations_counter
                    mutations_counter = 0

                    # Impose parameter space constrains using penalty
                    loss = loss + r1 * probability_domain_penalty(model.phase2.replication_prob)
                    loss = loss + r1 * probability_domain_penalty(model.phase1.motifs_prob)
                    loss = loss + r1 * probability_domain_penalty(model.phase2.lp_ber.profile)
                    loss = loss + r2 * probability_sum_penalty(model.phase1.motifs_prob)
                    loss = loss + r2 * probability_sum_penalty(model.phase2.lp_ber.profile)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Prints
                    print(f'Learning step: {step_counter}, Batch: {batch_counter} / {batches_this_step}, Loss: {loss}')

                    print('----------------------------------------------------------------------')
                    for name, param in model.named_parameters():
                        print(f'{name}: {param}')
                    print('----------------------------------------------------------------------')

                    # Log
                    log_line = {'loss': [loss.data.detach().item()]}
                    for name, param in model.named_parameters():
                        value = param.detach().numpy()
                        if len(param) > 1:
                            value = [value]
                        log_line[name] = value

                    pd.DataFrame(log_line).to_csv(log_csv, index=False, mode='a', header=not os.path.isfile(log_csv))

                    
                    # Reset loss
                    loss = torch.tensor([0.], requires_grad=True)

                    # Exit loop condition
                    if batch_counter == batches_this_step:
                        break





