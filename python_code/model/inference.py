import torch
import torch.nn.functional as F
from python_code.data_preprocess.count_mutations import mismatch_positions

# --- Hyperparameters --- #
epochs = 10
batch_size = 500
learning_rate = 1e-4
r1 = 100  # replication_prob panelty coefficient

def calc_target_likelihood(ancestor_nucleotide, descendant_nucleotide, target_probs):
    likelihood = torch.tensor([0.], requires_grad=True) 
    if ancestor_nucleotide == 'C':
        likelihood = likelihood + target_probs[1]  # short-patch ber
        if descendant_nucleotide == 'T':
            likelihood = likelihood + target_probs[0]  # replication
    return -torch.log(likelihood)

def calc_sequence_likelihood(ancestor_sequence, descendant_sequence, probs):
    likelihood = torch.tensor([0.], requires_grad=True) 
    targets = mismatch_positions(ancestor_sequence, descendant_sequence)
    for t in targets:
        likelihood = likelihood + calc_target_likelihood(ancestor_sequence[t], 
                                                         descendant_sequence[t],
                                                         probs[:, t])
    return likelihood

def probability_domain_penalty(prob):
    below_zero_penalty = F.relu(0 - prob) ** 2
    over_one_penalty = F.relu(prob - 1) ** 2
    return below_zero_penalty + over_one_penalty


def inference(model, data, ancestor_column='ancestor_alignment', descendant_column='sequence_alinment'):

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    batches_each_epoch = int(len(data) / batch_size)

    for epoch_counter in range(epochs):
        loss = torch.tensor([0.], requires_grad=True)
        data = data.sample(frac=1, ignore_index=True)  # Shuffle samples

        for sample_counter, row in data.iterrows():

            # Compute likelihood and accumulate loss
            probs = model(row[ancestor_column])
            loss = loss + calc_sequence_likelihood(row[ancestor_column], row[descendant_column], probs)
            
            if (sample_counter % batch_size == 0) & (sample_counter > 0):

                # Correct loss magnitude by batch size
                loss = loss / batch_size

                # Prints
                batch = int(sample_counter / batch_size)
                print(f'Epoch: {epoch_counter}, Batch: {batch} / {batches_each_epoch}, Loss: {loss}')
                

                # Impose parameter space constrains using penalty
                loss = loss + r1 * probability_domain_penalty(model.phase2.replication_prob)

                # Backpropagation
                loss.backward()
                optimizer.step()
                print(f' -------- > {[x for x in model.parameters()][0].data}')

                loss = torch.tensor([0.], requires_grad=True)


