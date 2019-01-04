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
        
        # add embedding layer after 
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features) # batch_size, embed_size
        
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, drop_prob=0.4):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # layer that turns words (represented by an integer -  vocabulary index)
        # into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # lstm takes takes word in embedded space - a vector(0.. embed_size-1) 
        # and outputs hidden states - vector (0 .. hidden_size-1)
        # need to use batch_first to avoid having to use the permute operation
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # output linear layer, scores for the various words
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.init_weights()

    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''

        # initialize embeddings
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1) 
        
    def init_hidden(self, batch_size):
        ''' At the start of training, we need to initialize a hidden state;
           there will be none because the hidden state is formed based on perviously seen data.
           So, this function defines a hidden state with all zeroes and of a specified size.'''
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
       
    
    # this link was useful to understand the magic that needs to happen here
    # https://discuss.pytorch.org/t/dynamic-parameter-declaration-in-forward-function/427/3
    def forward(self, features, captions):
        
        batch_size = features.shape[0]
        captions_len = captions.shape[1]
        
        # reset hidden states
        hidden = self.init_hidden(batch_size)
        
        # compute the rest of the embeddings
        # (excluding <end>, which is the expected output for the last step but never passed as input)
        embeds = self.word_embeddings( captions[0:batch_size, 0:(captions_len-1)] ) # batch_size, captions_len, embed_size
        
        # put the image features (in embedding space) at the start of the input vector for LSTM
        # concat along captions_len dimension
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        
        # lstm expects captions_len, batch_size, embed_size
        #for some reason I need to call .cuda here, not sure why? Maybe not neccessary any more
        # I trained this on a private machine, maybe it is caused by a different pytorch version
        lstm_out, hidden = self.lstm(embeds.cuda(), (hidden[0].cuda(), hidden[1].cuda()))
        
        # chain the operations so I don't need to declare more temporary matrices 
        outputs = self.fc(self.dropout(lstm_out))
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # what is states for?
        
        word_indexes = []
        
        # forget everything
        hidden = self.init_hidden(batch_size=1)
        
        
        lstm_out, hidden = self.lstm(inputs.cuda(), (hidden[0].cuda(), hidden[1].cuda()))
        
        outputs = self.fc(lstm_out)
       
        
        # get the index of the highest output from the network
        word_indexes.append(int(torch.max(
                                    outputs[0:1, 0:1, 0:self.vocab_size].squeeze(),
                                    0
                                   )[1]
                        ))
        # got the initial predicition, it should be 0 / <start>
        
        for i in range(0, max_len):
            
            # output form the first step is input for the next step
            embeds = self.word_embeddings(torch.cuda.LongTensor( [[ word_indexes[i] ]] ))
            
            lstm_out, hidden = self.lstm(embeds.cuda(), (hidden[0].cuda(), hidden[1].cuda()))
        
            outputs = self.fc(lstm_out)
            
            # get the index of the highest output from the network
            word_indexes.append(int(torch.max(
                                        outputs[0:1, 0:1, 0:self.vocab_size].squeeze(),
                                        0
                                       )[1]
                            ))
            
            # check if we predicted <end>, if yes then stop
            if (word_indexes[i+1] == 1):
                break
        
        return word_indexes # not tested yet
