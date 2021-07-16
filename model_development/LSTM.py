class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer,hidden_mlp, n_classes):
        super(LSTM,self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first = True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_mlp),nn.Tanh(),
                               nn.Linear(hidden_mlp, n_classes), nn.Sigmoid())
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, :, :]
        x = self.classifier(x)
        return x
