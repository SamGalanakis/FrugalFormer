import copy
import time
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.module import T
from dataprep import wiki_dataset,get_batch
from models import TransformerModel
from utils import generate_square_subsequent_mask
import wandb 


config_path = 'configs/baseline.yaml'
wandb.init(project='FrugalFormer', entity='samme013',config=config_path)
config = wandb.config


device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_data,test_data,val_data,vocab = wiki_dataset(config)
ntokens = len(vocab)



model = TransformerModel(ntokens, config['emsize'], config['nhead'], config['d_hidden'],
config['nlayers'], config['dropout']).to(device)



criterion = nn.CrossEntropyLoss()
lr = config['lr']  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(config['bptt']).to(device)

    num_batches = len(train_data) // config['bptt']
    for batch, i in enumerate(range(0, train_data.size(0) - 1, config['bptt'])):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != config['bptt']:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
     
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(config['bptt']).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, config['bptt']):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != config['bptt']:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)




best_val_loss = float('inf')
epochs = config['epochs']
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()