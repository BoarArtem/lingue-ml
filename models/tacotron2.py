import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PreNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 0.5 dropout rate - important!

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=True)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=True)

        return x

class PostNet(nn.Module):
    def __init__(self, mel_size):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=mel_size, out_channels=512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.Tanh()
            ),

            nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.Tanh()
            ),

            nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.Tanh()
            ),

            nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.Tanh()
            ),

            nn.Sequential(
                nn.Conv1d(in_channels=512, out_channels=mel_size, kernel_size=5, padding=2),
                nn.BatchNorm1d(mel_size)
            )
        ])

    def forward(self, x):
        # T - time
        # x: [B, T, mel_size]
        x = x.transpose(1, 2) # -> [B, mel_size, T]

        for conv in self.convs:
            x = conv(x)

        x = x.transpose(1, 2) # reverse -> [B, T, mel_size]
        return x


class LocalSensitiveAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # s_t (decoder_lstm_hidden_size) - [B, 1024]
        # m_i (encoder_bilstm_output) - [B, T, 512]
        # alpha_t-1 (prev_step_attention_weights) - [B, T]
        # f_i (location_features) - [B, T, 32]

        self.query = nn.Linear(in_features=1024, out_features=128, bias=False) # W_q * s_t in R^128
        self.memory = nn.Linear(in_features=512, out_features=128, bias=False) # W_m * m_i in R^128
        self.location_features = nn.Linear(in_features=32, out_features=128, bias=False) # W_f * f_i in R^128
        self.location_filter = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=31, padding=15)

        # f_i = F (1d conv) * alpha_t-1

        # alpha_t-1 [B, T] -(unsqueeze)> [B, 1, T] -> Conv1d(1, 32, kernel_size=31, padding=15) ->
        # -> [B, 32, T] -(transpose)> [B, T, 32]

        # e_t,i = wT * tanh(W_q * s_t + W_m * m_i + W_f * f_i)

        self.energy_score = nn.Linear(in_features=128, out_features=1, bias=False)

        self.bias = nn.Parameter(torch.zeros(128))

        # alpha_t,i (current_attention_weights) = softmax(e_t,i)

        # c_t = sum(alpha_t,i * m_i)

    def forward(self, decoder_lstm_hidden, encoder_bilstm_outputs, prev_attention_weights):
        f_i_step1 = prev_attention_weights.unsqueeze(1)
        f_i_step2 = self.location_filter(f_i_step1)
        f_i = f_i_step2.transpose(1, 2)

        query = self.query(decoder_lstm_hidden)
        memory = self.memory(encoder_bilstm_outputs)
        location = self.location_features(f_i)

        x = query + memory + location + self.bias
        computed_energy_score = self.energy_score(F.tanh(x))

        new_attention_weights = F.softmax(computed_energy_score, dim=1)

        context_vector = torch.bmm(new_attention_weights.unsqueeze(1), memory)
        # torch.bmm(input, mat2) - performs a batch matrix-matrix multiplication of input and mat2,
        # resulting in a batch of matrices

        return context_vector.squeeze(1), new_attention_weights

class Tacotron2Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, voc_size=10000, num_layers=1):
        super().__init__()
        self.character_embedding = nn.Embedding(voc_size, input_size)
        self.conv1d_combo = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
        )
        self.bi_directional_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        embedded = self.character_embedding(x)
        conv_out = self.conv1d_combo(embedded)
        _, (hidden, _) = self.bi_directional_lstm(conv_out)

        return hidden

class Tacotron2Decoder(nn.Module):
    def __init__(self, mel_size, hidden_size=512, voc_size=10000, num_layers=2):
        super().__init__()

        self.prenet = PreNet(mel_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.local_sensitive_attention = LocalSensitiveAttention()
        self.postnet = PostNet(mel_size)

        self.linear_projection = nn.Linear(in_features=hidden_size, out_features=mel_size)

    def forward(self, memory, prev_mel_frame, prev_attention_weights):
        pass







