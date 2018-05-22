import torch
from torch import nn
# spell
# num layers : 3
# hidden size : 512
# input size : 45 (because output y size is 45)

# Watch
# num layer : 3
# hidden size : 256
# input size : 512

class WLSNet(nn.Module):

    def __init__(self, num_layers=3, input_size=512, hidden_size=256):
        super(WLSNet, self).__init__()
        self.watch = Watch(num_layers, input_size, hidden_size)
        self.listen = Listen(num_layers, hidden_size)
        self.spell = Spell(num_layers, 45, hidden_size*2)

    def forward(self, audio, video):
        watch_outputs, watch_state = self.watch(video)
        listen_outputs, listen_state = self.listen(audio)
        return self.spell(watch_outputs, listen_outputs ,watch_state, listen_state)

class Watch(nn.Module):
    '''
    layer size 3
    cell size 256
    '''
    def __init__(self, num_layers, input_size, hidden_size):
        super(Watch, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : 3-D torch Tensor
            (batch_size, sequence, features) 
        '''
        outputs, states = self.lstm(x)

        return (outputs, states[0])

class Listen(nn.Module):
    '''
    layer size 3
    cell size 256
    '''
    def __init__(self, num_layers, hidden_size):
        super(Listen, self).__init__()
        self.lstm = nn.LSTM(13, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : 3-D torch Tensor
            (batch_size, sequence, features) 
        '''
        outputs, states = self.lstm(x)

        return (outputs, states[0])

class Spell(nn.Module):
    def __init__(self, num_layers=3, input_size=45, hidden_size=512):
        super(Spell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionVideo = Attention(hidden_size+256)
        self.attentionAudio = Attention(hidden_size+256)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size*2, 45),
            nn.Softmax()
        )
        
    def forward(self, watch_outputs, listen_outputs, watch_state, listen_state):
        '''
        1. initialize hidden_state with watch_state and listen_state
            (num_layers, batch_size, hidden_size(256 + 256))
        2. initialize cell_state
            (num_layers, batch_size, hidden_size)
        3. initialize input_y (first y is zero)
            (batch_size, 1, input_size)
        each sequence step
            except i == 0
            1. compute attention hidden state(video)
                (batch_size, num_layers, feature dimension(256))
            2. compute attention hidden state(video)
                (batch_size, num_layers, feature dimension(256))
            3. concatenate
                (batch_size, num_layers, feature dimension(512))
            4. input_y reshape
                (batch_size, input_size)
            5. sum to delete
                (batch_size, feature dimension(512))
            6. concatenate to go through mlp
                (batch_size, input_size + feature dimension(512))
            7. unsqueeze input_y to make sequence dimension

        Return
        ------
        outputs : 3-D torch Tensor
            (batch_size, output_sequence, 45(self.input_size))
        '''
        batch_size = watch_outputs.size()[0]
        hidden_state = torch.cat([watch_state, listen_state], dim=2).permute(1, 0, 2)
        cell_state = torch.zeros(self.num_layers, watch_outputs.size()[0], self.hidden_size)
        input_y = torch.zeros(watch_outputs.size()[0], 1, self.input_size)
        outputs = []

        #이부분은 그냥 임의의 숫자로 함
        for i in range(5):
            input_y, (hidden_state, cell_state) = self.lstm(input_y, (hidden_state.permute(1, 0, 2), cell_state))
            video_hidden_state = self.attentionVideo(hidden_state, watch_outputs)
            audio_hidden_state = self.attentionVideo(hidden_state, listen_outputs)
            hidden_state = torch.cat([video_hidden_state, audio_hidden_state], dim=2)
            reshaped_input_y = input_y.view(batch_size, -1)
            sum_hidden_state = torch.sum(hidden_state, dim=1)
            input_y = self.mlp(torch.cat([reshaped_input_y, sum_hidden_state], dim=1)).unsqueeze(1)
            outputs.append(input_y)
            #if end break and return
        return torch.cat(outputs, dim=1)

class Attention(nn.Module):

    def __init__(self, cat_feature_size):
        super(Attention, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(cat_feature_size, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=2)

    # def forward(self, prev_hidden_state, annotations):
    #     '''
    #     expand to concatenate

    #     Parameters
    #     ----------
    #     prev_hidden_state : 2-D torch Tensor
    #         (batch_size, feature dimension(default 512))
        
    #     annotations : 3-D torch Tensor
    #         (batch_size, sequence_length, feature dimension(256))

    #     Return
    #     ------
    #     context : 3-D torch Tensor
    #         (batch_size, feature dimension(256))
    #     '''
    #     batch_size = prev_hidden_state.size()[0]
    #     prev_hidden_state = prev_hidden_state.expand(batch_size, annotations.size()[1], prev_hidden_state.size()[1])
    #     concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
    #     energy = self.softmax(self.dense(concatenated).view(batch_size, -1)).unsqueeze(2)

    #     return torch.sum(energy * annotations, dim=1)

    def forward(self, prev_hidden_state, annotations):
        '''

        1. expand prev_hidden_state dimension
            (batch_size, num_layers, sequence_length, feature dimension(512)) 
        2. expand annotations dimension
            (batch_size, num_layers, sequence_length, feature dimension(256))
        3. concatenate
            (batch_size, num_layers, sequence_length, feature dimension(512) + feature dimension(256))
        4. dense
            (batch_size, num_layers, sequence_length, 1)
        5. softmax
            (batch_size, num_layers, sequence_length, 1)
        6. expand and element wise multiplication(annotation and alpha)
            (batch_size, num_layers, sequence_length, feature dimension(256))
        7. sum
            (batch_size, num_layers, feature dimension(256))


        Parameters
        ----------
        prev_hidden_state : 3-D torch Tensor
            (num_layers, batch_size, feature dimension(default 512))
        
        annotations : 3-D torch Tensor
            (batch_size, sequence_length, feature dimension(256))

        Return
        ------
        context : 3-D torch Tensor
            (batch_size, num_layers, feature dimension(256))
        '''
        prev_hidden_state = prev_hidden_state.permute(1, 0, 2)

        batch_size, num_layers, feature_state = prev_hidden_state.size()
        _, sequence_length, feature_annotation = annotations.size()
        
        prev_hidden_state = prev_hidden_state.unsqueeze(2).expand(batch_size, num_layers, sequence_length, feature_state)
        annotations = annotations.unsqueeze(1).expand(batch_size, num_layers, sequence_length, feature_annotation)

        concatenated = torch.cat([prev_hidden_state, annotations], dim=3)
        energy = self.dense(concatenated)
        alpha = self.softmax(energy)
        alpha = alpha.expand_as(annotations)

        return torch.sum(alpha * annotations, dim=2)

class Encoder(nn.Module):
    '''modified VGG-M
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 96, (7, 7), (2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0), ceil_mode=True)
        )
        
        self.fc = nn.Linear(4608, 512)

    def forward(self, x):
        return self.fc(self.encoder(x).view(x.size(0), -1))