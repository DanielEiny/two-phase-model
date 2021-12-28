import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from python_code.data_preprocess.count_mutations import mismatch_positions

# --- Hyperparameters --- #
epochs = 3
batch_size_schedule = [100, 1000, 10000]
batches_each_epoch_schedule = [100, 100, 1000]
learning_rate_schedule = [1e-2, 1e-2, 1e-2]
r1 = 100  # probability domain panelty coefficient
r2 = 10  # probability sum panelty coefficient
log_path = 'results/model/log'

def calc_target_likelihood(ancestor_nucleotide, descendant_nucleotide, targeting_prob, replication_prob):
    likelihood = torch.tensor([0.], requires_grad=True) 
    if ancestor_nucleotide == 'C':  # short-patch ber
        G_or_T_or_A_prob = 1 / 3
        likelihood = likelihood + targeting_prob * (1 - replication_prob) * G_or_T_or_A_prob
        if descendant_nucleotide == 'T':  # Possibly replication
            likelihood = likelihood + targeting_prob * replication_prob
    return -torch.log(likelihood)

def calc_sequence_likelihood(ancestor_sequence, descendant_sequence, targeting_probs, replication_probs):
    likelihood = torch.tensor([0.], requires_grad=True) 
    targets = mismatch_positions(ancestor_sequence, descendant_sequence)
    for t in targets:
        likelihood = likelihood + calc_target_likelihood(ancestor_sequence[t], 
                                                         descendant_sequence[t],
                                                         targeting_probs[t],
                                                         replication_probs[t])
    return likelihood, len(targets)

def probability_domain_penalty(prob):
    below_zero_penalty = F.relu(0 - prob) ** 2
    over_one_penalty = F.relu(prob - 1) ** 2
    return below_zero_penalty + over_one_penalty

def probability_sum_penalty(probs):
    return (probs.sum() - 1) ** 2

def inference(model, data, ancestor_column='ancestor_alignment', descendant_column='sequence_alinment'):
    now = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
    log_dir = os.path.join(log_path, now)
    os.makedirs(log_dir)
    log_csv = os.path.join(log_dir, 'log.csv')

    for epoch_counter in range(epochs):

        batch_size = batch_size_schedule[epoch_counter]
        learning_rate = learning_rate_schedule[epoch_counter]
        batches_each_epoch = batches_each_epoch_schedule[epoch_counter]

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        #batches_each_epoch = int(len(data) / batch_size)

        loss = torch.tensor([0.], requires_grad=True)
        data = data.sample(frac=1, ignore_index=True)  # Shuffle samples

        mutations_counter = 0

        for sample_counter, row in data.iterrows():

            # Compute likelihood and accumulate loss
            targeting_probs, replication_probs = model(row[ancestor_column])
            negative_log_likelihood, n_mutations = calc_sequence_likelihood(row[ancestor_column], row[descendant_column], targeting_probs, replication_probs)
            loss = loss + negative_log_likelihood
            mutations_counter += n_mutations
            
            if (sample_counter % batch_size == 0) & (sample_counter > 0):

                # Scale loss magnitude by the number of mutations
                loss = loss / mutations_counter
                mutations_counter = 0

                # Prints
                batch = int(sample_counter / batch_size)
                print(f'Epoch: {epoch_counter}, Batch: {batch} / {batches_each_epoch}, Loss: {loss}')

                # Log
                log_line = pd.DataFrame({'loss': [loss.data.detach().item()], 'param': [model.phase2.replication_prob.data.detach().item()]})
                log_line.to_csv(log_csv, index=False, mode='a', header=not os.path.isfile(log_csv))

                # Impose parameter space constrains using penalty
                loss = loss + r1 * probability_domain_penalty(model.phase2.replication_prob)
                loss = loss + r1 * probability_domain_penalty(model.phase1.motifs_prob[0])
                loss = loss + r2 * probability_sum_penalty(model.phase1.motifs_prob)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = torch.tensor([0.], requires_grad=True)
                
                print('---------------------------------------------------------')
                for name, param in model.named_parameters():
                    print(f'{name}: {param}')
                print('---------------------------------------------------------')

                if batch == batches_each_epoch:
                    break





