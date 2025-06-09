import torch
import tiktoken
from chapter5.LossLoader import calc_loss_batch, calc_loss_loader
from chapter5.TokenUtilities import text_to_token_ids, token_ids_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")


def evaluate_model(model,
                   train_loader,
                   val_loader,
                   device,
                   eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# token_ids = generate( model=model,
#                      idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
#                      max_new_tokens=26,
#                      context_size=BASE_CONFIG["context_length"],
#                      top_k=50,
#                      temperature=1.5)

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model=model,
                             idx=encoded,
                             max_new_tokens=50,
                             context_size=context_size,
                             top_k=50,
                             temperature=1.5)
    #   token_ids = generate_text_simple( model=model,
    #                                     idx=encoded,
    #                                     max_new_tokens=50,
    #                                     context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       device,
                       num_epochs,
                       eval_freq,
                       eval_iter,
                       start_context,
                       tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model,
                                                      train_loader,
                                                      val_loader,
                                                      device,
                                                      eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch + 1} Step {global_step:06d}: "
                      f"Train loss {train_loss:.3f} "
                      f"Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


# token_ids = generate( model=model,
#                      idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
#                      max_new_tokens=26,
#                      context_size=BASE_CONFIG["context_length"],
#                      top_k=50,
#                      temperature=1.5)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(torch.tensor(logits < min_val), torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx