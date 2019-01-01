import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        print(resnet)
        #modules = list(resnet.children())[:-3] # 1024 x 14 x14
        #modules = list(resnet.children())[:-2] # 512 x 7 x 7
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # add embedding layer after 
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        # add batch normalization?
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        print("encoder ", features.shape)
        features = features.view(features.size(0), -1)
        print("encoder ", features.shape)
        
        features = self.embed(features) # batch_size, embed_size
        
        print("encoder ", images.shape, features.shape)
        #features = self.bn(features)
        return features
    
class AttentionNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        print(resnet)
        #modules = list(resnet.children())[:-3] # 1024 x 14 x14
        #modules = list(resnet.children())[:-2] # 512 x 7 x 7
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # add embedding layer after 
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        # add batch normalization?
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        print("encoder ", features.shape)
        features = features.view(features.size(0), -1)
        print("encoder ", features.shape)
        
        features = self.embed(features) # batch_size, embed_size
        
        print("encoder ", images.shape, features.shape)
        #features = self.bn(features)
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
        #self.hidden = self.init_hidden()
        
        # initialize the weights
        self.init_weights()
        
        self.max_captions_len = max_captions_len # actually it is 57, but this is more round
        self.max_batch_size = max_batch_size
        
        #self.outputs = torch.zeros(batch_size, captions_len, self.vocab_size)
        
        #self.embeds = torch.zeros(self.max_batch_size, self.max_captions_len, self.embed_size)
    
        #self.lstm_out = torch.zeros(self.max_batch_size, self.max_captions_len, self.hidden_size)
    
        #self.outputs = torch.zeros(self.max_captions_len, self.max_batch_size, self.vocab_size)
    
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
        
        #print("batch_size: ", batch_size)
        
        #print("feat unsqueezed", features.unsqueeze(1).shape)
        
        # compute the rest of the embeddings
        # (excluding <end>, which is the expected output for the last step but never passed as input)
        embeds = self.word_embeddings( captions[0:batch_size, 0:(captions_len-1)] ) # batch_size, captions_len, embed_size
        
        # put the image features (in embedding space) at the start of the input vector for LSTM
        # concat along captions_len dimension
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        
        
        # lstm expects captions_len, batch_size, embed_size
        #for some reason I need to call .cuda here, I guess permute doesn't work in GPU memory?
        lstm_out, hidden = self.lstm(embeds.permute(1,0,2).cuda())
        
        # chain the operations so I don't need to declare more temporary matrices 
        outputs = self.log_softmax(
                                       self.fc(
                                           self.dropout(lstm_out)
                                       )
                                    )
        ret = outputs[0:captions_len, 0:batch_size, 0:self.vocab_size].permute(1,0,2)
        
        print("ret", ret.shape, ret[0, 0, :])
        # discard what we don't need to return, otherwise the automatic checks complain
        return ret

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # what is states for?
        
        word_indexes = []
        
        # forget everything
        hidden = self.init_hidden(batch_size=1)
        
        # this will be (1, 1, embed_size)
        #inputs.unsqueeze_(0).unsqueeze_(0)
        
        
        lstm_out, hidden = self.lstm(inputs)
        
        outputs = self.log_softmax(
                               self.fc(lstm_out)
                            )
        
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
            
            #print(embeds.shape)
            
            lstm_out, hidden = self.lstm(embeds)
        
            outputs = self.log_softmax(
                                   self.fc(lstm_out)
                                )
            # get the index of the highest output from the network
            word_indexes.append(int(torch.max(
                                        outputs.squeeze(),
                                        0
                                       )[1]
                            ))
            
            # check if we predicted <end>, if yes then stop
            if (word_indexes[i+1] == 1):
                break
        
        return word_indexes # not tested yet
