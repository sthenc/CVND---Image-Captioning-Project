import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, drop_prob=0.1, max_batch_size=64, max_captions_len=64):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # layer that turns words (represented by an integer -  vocabulary index)
        # into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # lstm takes takes word in embedded space - a vector(0.. embed_size-1) 
        # and outputs hidden states - vector (0 .. hidden_size-1)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # output linear layer, scores for the various words
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # apply across vocab_size dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # init hidden states - contains tuple (h, c)
        self.hidden = self.init_hidden()
        
        # initialize the weights
        self.init_weights()
        
        self.max_captions_len = max_captions_len # actually it is 57, but this is more round
        self.max_batch_size = max_batch_size
        
        #self.outputs = torch.zeros(batch_size, captions_len, self.vocab_size)
        
        self.embeds = torch.zeros(self.max_batch_size, self.max_captions_len, self.embed_size)
    
        self.lstm_out = torch.zeros(self.max_batch_size, self.max_captions_len, self.hidden_size)
    
        self.outputs = torch.zeros(self.max_captions_len, self.max_batch_size, self.vocab_size)
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1) 
        
    def init_hidden(self):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))
       
    
    # this link was useful to understand the magic that needs to happen here
    # https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/3
    def forward(self, features, captions):
        
        batch_size = features.shape[0]
        captions_len = captions.shape[1]
        
        # reset hidden states
        self.hidden = self.init_hidden()
        
        #print("batch_size: ", batch_size)
        
        #print("feat unsqueezed", features.unsqueeze(1).shape)
        
        # put the image features (in embedding space) at the start of the input vector for LSTM
        self.embeds[:, 0, :] = features.unsqueeze(1)[0:batch_size, 0, 0:self.embed_size]
        
        # compute the rest of the embeddings
        # (excluding <end>, which is the expected output for the last step but never passed as input)
        self.embeds[:, 1:captions_len, :] = self.word_embeddings(
                                                                 captions[0:batch_size, 0:(captions_len-1)]
                                                                )
        
        # lstm expects captions_len, batch_size, embed_size
        #for some reason I need to call .cuda here, I guess permute doesn't work in GPU memory?
        self.lstm_out, self.hidden = self.lstm(self.embeds.permute(1,0,2).cuda())
        
        # chain the operations so I don't need to declare more temporary matrices 
        self.outputs = self.log_softmax(
                                       self.fc(
                                           self.dropout(self.lstm_out)
                                       )
                                    )
        
        # discard what we don't need to return, otherwise the automatic checks complain
        return self.outputs[0:captions_len, 0:batch_size, 0:self.vocab_size].permute(1,0,2)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # what is states for?
        
        # forget everything
        self.hidden = self.init_hidden()
        
        # this will be (1, 1, embed_size)
        inputs.unsqueeze_(0).unsqueeze_(0)
        
        self.lstm_out, self.hidden = self.lstm(inputs)
        
        self.outputs = self.log_softmax(
                               self.fc(self.lstm_out)
                            )
        
        word_index = int(torch.max(
                                    self.outputs[0:1, 0:1, 0:vocab_size].squeeze(),
                                    0
                                   )[1]
                        )
        
        return word_index # not tested yet