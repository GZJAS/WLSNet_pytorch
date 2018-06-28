import torch
from torch import nn, optim
from dataLoader import get_dataloaders
from model import Watch, Spell
from charSet import get_charSet, init_charSet
import argparse
import yaml
import Levenshtein as Lev

torch.backends.cudnn.enabled = True

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, help='the number of iterations')
    parser.add_argument('--vmax', type=int, help='video tensor max length')
    parser.add_argument('--tmax', type=int, help='text tensor max length')
    parser.add_argument('--path', type=str, help='data path')
    parser.add_argument('--bs', type=int, help='batch size', default=2)
    parser.add_argument('--worker', type=int, help='the number of data loader worker', default=0)
    parser.add_argument('--validation_ratio', type=float, help='validation ratio', default=0.0001)
    parser.add_argument('--learning_rate_decay_epoch', type=int, help='learning rate decay per certain epoch')
    parser.add_argument('--learning_rate_decay_ratio', type=float, help='learning rate decay ratio')
    parser.add_argument('--save_every', type=int, help='save every certain epoch', default=30)
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--language', type=str, help='language')
    parser.add_argument('--hidden_size', type=int, help='hidden dimension size', default=512)
    parser.add_argument('--layer_size', type=int, help='layer size', default=3)

    return parser.parse_args()

def train(input_tensor, length_tensor, target_tensor,
        watch, spell, 
        watch_optimizer, spell_optimizer, 
        criterion, is_train):
    '''
    train or validate the model

    1.
    check cuda and init device var(see pytorch 0.4 migration)
    init results var to calculate cer(character error rate)
    load vars to GPU if available

    2.
    optimizer set zero_grad

    3.
    get label's sentence size to divide loss
    set cer and loss to zero

    4.
    get through watch model(encoder)
        watch_outputs : 3-D torch tensor
            collect RNN's outputs
            size (batch_size, sequence_length, hidden_size)
        watch_state : 3-D torch tensor
            first layer of RNN's hidden state
            size (layer_size, batch_size, hidden_size)

    5.
    init inputs
        spell_input : 3-D torch tensor
            fill with <sos>(start of sentence) for first step
            size (batch_size, 1, hidden_size)
        spell_hidden : 3-D torch tensor
            first step's hidden state input is watch model's hidden state
        cell_state : 3-D torch tensor
            size is same as spell_hidden
        context : 3-D torch tensor
            attention context matrix
            size (batch_size, 1, hidden_size)

    6.
    get through spell model(encoder)
        in this model, previous output is present input.
        so, we put inputs step by step

        spell_output
            size(batch_size, 1, get_charSet().get_total_num())
        topk = top k number
            topk(1, dim=2) means take the biggest number within all characters
        topi = top index
            topi size (batch_size, 1, 1)

        if train
            right previous label becomes present input
        else
            previous output becomes present input

    Parameters
    ----------
    input_tensor : 4-D torch tensor
        sequential grey image data(video)
        size (batch_size, sequence_length, 120, 120)

    length_tensor : 2-D torch tensor
        sequence length of each batch
        size (batch_size, 1)

    target_tensor : 2-D torch tensor
        sentence data(0 ~ get_charSet().get_total_num()-1)
        size (batch_size, 1)

    watch : watch model

    spell : spell model

    watch_optimizer : optimizer for watch model

    spell_optimizer : optimizer for spell model

    criterion : loss function

    is_train : boolean
        if it is train of inference
    '''

    # 1. block
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    length_tensor = length_tensor.to(device)

    # 2. block
    watch_optimizer.zero_grad()
    spell_optimizer.zero_grad()

    # 3. block
    target_length = target_tensor.size(1)
    loss = 0
    cer = 0

    # 4. block
    watch_outputs, watch_state = watch(input_tensor, length_tensor)

    # 5. block
    spell_input = torch.tensor([[get_charSet().get_index_of('<sos>')]]).repeat(watch_outputs.size(0), 1).to(device)
    spell_hidden = watch_state
    cell_state = torch.zeros_like(spell_hidden).to(device)
    context = torch.zeros(watch_outputs.size(0), 1, spell_hidden.size(2)).to(device)

    # 6. block
    if is_train:
        for di in range(target_length):
            spell_output, spell_hidden, cell_state, context = spell(spell_input, spell_hidden, cell_state, watch_outputs, context)
            _, topi = spell_output.topk(1, dim=2)
            spell_input = target_tensor[:, di].long().unsqueeze(1)
            
            loss += criterion(spell_output.squeeze(1), target_tensor[:, di].long())
            results.append(topi.cpu().squeeze(1))
        loss = loss.to(device)
        loss.backward()

        watch_optimizer.step()
        spell_optimizer.step()

    else:
        for di in range(target_length):
            spell_output, spell_hidden, cell_state, context = spell(
                spell_input, spell_hidden, cell_state, watch_outputs, context)
            _, topi = spell_output.topk(1, dim=2)
            spell_input = topi.squeeze(1).detach()
            
            if int(target_tensor[0, di]) != get_charSet().get_index_of('<pad>'):
                print('output : ', get_charSet().get_char_of(int(topi.squeeze(1)[0])), 'label : ', get_charSet().get_char_of(int(target_tensor[0, di])))

            loss += criterion(spell_output.squeeze(1), target_tensor[:, di].long())
            results.append(topi.cpu().squeeze(1))
    
    results = torch.cat(results, dim=1)
    for batch in range(results.size(0)):
        output = ''
        label = ''
        for index in range(target_length):
            output += get_charSet().get_char_of(int(results[batch, index]))
            label += get_charSet().get_char_of(int(target_tensor[batch, index]))
        label = label.replace('<pad>', '').replace('<eos>', '@')
        output = output.replace('<eos>', '@')[:output.find('@')].replace('<pad>', '$').replace('<sos>', '&')
        cer += Lev.distance(output, label)
    return loss.item() / target_length, cer

def trainIters(args):
    init_charSet(args['LANGUAGE'])

    watch = Watch(args['LAYER_SIZE'], args['HIDDEN_SIZE'], args['HIDDEN_SIZE'])
    spell = Spell(args['LAYER_SIZE'], args['HIDDEN_SIZE'], get_charSet().get_total_num())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    watch = watch.to(device)
    spell = spell.to(device)
    
    watch_optimizer = optim.Adam(watch.parameters(),
                    lr=args['LEARNING_RATE'])
    spell_optimizer = optim.Adam(spell.parameters(),
                    lr=args['LEARNING_RATE'])
    watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=args['LEARNING_RATE_DECAY_EPOCH'], gamma=args['LEARNING_RATE_DECAY_RATIO'])
    spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=args['LEARNING_RATE_DECAY_EPOCH'], gamma=args['LEARNING_RATE_DECAY_RATIO'])
    criterion = nn.CrossEntropyLoss(ignore_index=get_charSet().get_index_of('<pad>'))

    train_loader, eval_loader = get_dataloaders(args['PATH'], args['BS'], args['VMAX'], args['TMAX'], args['WORKER'], args['VALIDATION_RATIO'])

    total_batch = len(train_loader)
    total_eval_batch = len(eval_loader)

    for epoch in range(args['ITER']):
        avg_loss = 0.0
        avg_eval_loss = 0.0
        avg_cer = 0.0
        avg_eval_cer = 0.0
        watch_scheduler.step()
        spell_scheduler.step()

        watch = watch.train()
        spell = spell.train()

        for i, (data, length, labels) in enumerate(train_loader):

            loss, cer = train(data, length, labels,
                        watch, spell,
                        watch_optimizer, spell_optimizer,
                        criterion, True)
            avg_loss += loss
            avg_cer += cer
            print('Batch : ', i + 1, '/', total_batch, ', ERROR in this minibatch: ', loss)
            print('Character error rate : ', cer)
        
        watch = watch.eval()
        spell = spell.eval()

        for k, (data, length, labels) in enumerate(eval_loader):
            loss, cer = train(data, length, labels, watch, spell, watch_optimizer, spell_optimizer, criterion, False)
            avg_eval_loss += loss
            avg_eval_cer += cer
        print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
        print('epoch:', epoch, ' Average CER:', float(avg_cer/total_batch))
        print('epoch:', epoch, ' Validation_loss:', float(avg_eval_loss/total_eval_batch))
        print('epoch:', epoch, ' Average CER:', float(avg_eval_cer/total_eval_batch))
        if epoch % args['SAVE_EVERY'] == 0 and epoch != 0:
            torch.save(watch, 'watch{}.pt'.format(epoch))
            torch.save(spell, 'spell{}.pt'.format(epoch))

if __name__ == '__main__':
    with open('len100.yaml', 'r') as f:
        config = yaml.load(f)
    trainIters(config['CONFIG'])
