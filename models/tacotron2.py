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
    def __init__(self, mel_size=80, prenet_hidden=256, encoder_out=512, decoder_lstm_hidden=1024, num_layers=2):
        super().__init__()

        self.mel_size = mel_size
        self.decoder_lstm_hidden = decoder_lstm_hidden

        self.prenet = PreNet(mel_size, prenet_hidden)
        self.lstm = nn.LSTM(prenet_hidden + encoder_out, decoder_lstm_hidden, num_layers=num_layers, batch_first=True)

        self.attention = LocalSensitiveAttention()

        self.mel_projection = nn.Linear(in_features=decoder_lstm_hidden + encoder_out, out_features=mel_size)
        self.stop_projection = nn.Linear(in_features=decoder_lstm_hidden + encoder_out, out_features=1)

        self.postnet = PostNet(mel_size)

    def forward(self, encoder_outputs, mel_targets=None, max_steps=1000):
        # encoder outs - [B, T, 512]

        batch_size = encoder_outputs.size(0)
        t_enc = encoder_outputs.size(1)
        device = encoder_outputs.device

        # Initialize zero prev mels

        prev_mel = torch.zeros((batch_size, self.mel_size), device=device)
        prev_attention = torch.zeros((batch_size, t_enc), device=device)
        lstm_hidden = None

        mel_outputs, stop_outputs, attention_weights = [], [], []

        for t in range(max_steps):
            # PreNet
            prenet_out = self.prenet(prev_mel)

            # Attention: get context from encoder
            if lstm_hidden is not None:
                query = lstm_hidden[0][-1].unsqueeze(1) # [B, 1, 1024], decoder_lstm
            else:
                query = torch.zeros((batch_size, 1, self.decoder_lstm_hidden), device=device)

            context, prev_attention = self.attention(query, encoder_outputs, prev_attention)

            # LSTM
            lstm_input = torch.cat((prenet_out, context), dim=-1) # [B, 1, 768]
            lstm_out, lstm_hidden = self.lstm(lstm_input, lstm_hidden) # get lstm - [B, 1, 1024]

            # Projections: input = concat(lstm_out, context)
            proj_input = torch.cat((lstm_out, context), dim=-1) # [B, 1, 1536]
            mel_frame = self.mel_projection(proj_input) # [B, 1, 80]
            stop_token = self.stop_projection(proj_input)

            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_token)
            attention_weights.append(prev_attention)

            # Teacher forcing during training, autoregressive at inference
            if mel_targets is not None:
                prev_mel = mel_targets[:, t:t+1, :]
            else:
                prev_mel = mel_frame # apply new mel

                if torch.sigmoid(stop_token) > 0.5:
                    break

        # Stack all frames
        mel_outputs = torch.cat(mel_outputs, dim=1) # [B, T_dec, 80]
        stop_outputs = torch.cat(stop_outputs, dim=1) # [B, T_dec, 1]

        # PostNet residual refinement
        mel_outputs_post = mel_outputs + self.postnet(mel_outputs)

        return mel_outputs_post, mel_outputs, stop_outputs, attention_weights

class Tacotron2(nn.Module):
    def __init__(self, input_size, mel_size, prenet_hidden, encoder_out, decoder_lstm_hidden, num_layers):
        super().__init__()
        self.encoder = Tacotron2Encoder(input_size, mel_size, prenet_hidden, num_layers)
        self.decoder = Tacotron2Decoder(mel_size, prenet_hidden, encoder_out, decoder_lstm_hidden, num_layers)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        mel_outputs, mel_outputs_post, stop_outputs, attention_weights = self.decoder(encoder_outputs)




