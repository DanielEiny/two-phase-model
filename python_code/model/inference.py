import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
from python_code.data_preprocess.count_mutations import mismatch_positions

# --- Hyperparameters --- #
epochs = 100
batch_size = 5000
learning_rate = 1e-1
r1 = 100  # replication_prob panelty coefficient
log_path = 'results/model/log'

def calc_target_likelihood(ancestor_nucleotide, descendant_nucleotide, target_probs):
    likelihood = torch.tensor([0.], requires_grad=True) 
    if ancestor_nucleotide == 'C':  # short-patch ber
        G_or_T_or_A_prob = 1 / 3
        likelihood = likelihood + target_probs[1] * G_or_T_or_A_prob
        if descendant_nucleotide == 'T':  # replication
            likelihood = likelihood + target_probs[0]
    return -torch.log(likelihood)

def calc_sequence_likelihood(ancestor_sequence, descendant_sequence, probs):
    likelihood = torch.tensor([0.], requires_grad=True) 
    targets = mismatch_positions(ancestor_sequence, descendant_sequence)
    for t in targets:
        likelihood = likelihood + calc_target_likelihood(ancestor_sequence[t], 
                                                         descendant_sequence[t],
                                                         probs[:, t])
    # Average over all targets
    if len(targets) > 0:
        likelihood = likelihood / len(targets)
    return likelihood

def probability_domain_penalty(prob):
    below_zero_penalty = F.relu(0 - prob) ** 2
    over_one_penalty = F.relu(prob - 1) ** 2
    return below_zero_penalty + over_one_penalty


def inference(model, data, ancestor_column='ancestor_alignment', descendant_column='sequence_alinment'):
    now = datetime.now().strftime("%d_%m_%Y-%H:%M:%S")
    log_dir = os.path.join(log_path, now)
    os.makedirs(log_dir)
    log_csv = os.path.join(log_dir, 'log.csv')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    batches_each_epoch = int(len(data) / batch_size)

    for epoch_counter in range(epochs):
        loss = torch.tensor([0.], requires_grad=True)
        data = data.sample(frac=1, ignore_index=True)  # Shuffle samples

        for sample_counter, row in data.iterrows():

            # Compute likelihood and accumulate loss
            mp = mismatch_positions(row[ancestor_column], row[descendant_column])


            probs = model(row[ancestor_column])
            loss = loss + calc_sequence_likelihood(row[ancestor_column], row[descendant_column], probs)
            
            if (sample_counter % batch_size == 0) & (sample_counter > 0):

                # Scale loss magnitude by batch size
                loss = loss / batch_size

                # Prints
                batch = int(sample_counter / batch_size)
                print(f'Epoch: {epoch_counter}, Batch: {batch} / {batches_each_epoch}, Loss: {loss}')

                # Log
                log_line = pd.DataFrame({'loss': [loss.data.detach().item()], 'param': [model.phase2.replication_prob.data.detach().item()]})
                log_line.to_csv(log_csv, index=False, mode='a', header=not os.path.isfile(log_csv))

                # Impose parameter space constrains using penalty
                loss = loss + r1 * probability_domain_penalty(model.phase2.replication_prob)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = torch.tensor([0.], requires_grad=True)
                
                print('---------------------------------------------------------')
                for name, param in model.named_parameters():
                    print(f'{name}: {param}')
                print('---------------------------------------------------------')



