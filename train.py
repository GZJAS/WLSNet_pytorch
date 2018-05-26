import torch
from dataLoader import to_int

def train(watch_input_tensor, target_tensor,
        watch, listen, spell, 
        watch_optimizer, listen_optimizer, spell_optimizer, 
        criterion, listen_input_tensor=None):

    watch_optimizer.zero_grad()
    if listen_input_tensor:
        listen_optimizer.zero_grad()
    spell_optimizer.zero_grad()

    target_length = target_tensor.size(1)

    loss = 0

    watch_outputs, watch_state = watch(watch_input_tensor)

    if listen_input_tensor:
        listen_outputs, listen_state = listen(listen_input_tensor)
    else:
        listen_state = torch.zeros_like(watch_state)
        listen_outputs = torch.zeros(watch_outputs.size(0), 1, watch_outputs.size(2))
    #sos token
    spell_input = torch.tensor([[37]]).repeat(watch_outputs.size(0), 1)
    spell_hidden = torch.cat([watch_state, listen_state], dim=2)
    cell_state = torch.zeros_like(spell_hidden)
    context = torch.zeros(watch_outputs.size(0), 1, spell_hidden.size(2))

    # test = [spell_hidden]
    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):
        spell_output, spell_hidden, cell_state, context = spell(
            spell_input, spell_hidden, cell_state, watch_outputs, listen_outputs, context)
        topv, topi = spell_output.topk(1, dim=2)
        spell_input = target_tensor[:, di].long().unsqueeze(1)
        
        #print(topi.squeeze(1).detach().size())  # detach from history as input
        # print(spell_output)
        # print(spell_input)
        print(to_int[int(topi.squeeze(1)[0])])
        # print(torch.equal(test[-1], spell_hidden))
        # test.append(spell_hidden)
        loss += criterion(spell_output.squeeze(1), target_tensor[:, di].long())

    loss.backward()

    watch_optimizer.step()
    if listen_input_tensor:
        listen_optimizer.step()
    spell_optimizer.step()

    return loss.item() / target_length