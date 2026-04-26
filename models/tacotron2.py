import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.ljspeech import LJSpeechDataset, get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
import os
from pathlib import Path

# Add project root to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

class PreNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # 0.5 dropout rate - important!

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)

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

        # torch.bmm(input, mat2) - performs a batch matrix-matrix multiplication of input and mat2,
        # resulting in a batch of matrices
        # [B, 1, T] @ [B, T, 512] -> [B, 1, 512]
        context_vector = torch.bmm(new_attention_weights.transpose(1, 2), encoder_bilstm_outputs)

        return context_vector, new_attention_weights.squeeze(-1)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=self.padding,
                              dilation=dilation)

    def forward(self, x):
        # causal conv - ensures no future context is used in the convolution
        out = self.conv(x)
        return out[:, :, :-self.padding] # remove future context

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size=2, dilation=dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size=2, dilation=dilation)

        self.residual = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # dilated conv -> tanh(dilated_conv) and sigmoid(dilated_conv) multiplication -> shape (1x1)
        # -> apply residual and out and use skip connections for 1x1
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))

        out = filter_out * gate_out

        skip = self.skip(out)
        residual = self.residual(x)

        # residual - take prev output and add that to new one
        # and skip output - add skip output to next layer
        return x + residual, skip

class WaveNet(nn.Module):
    def __init__(self, mel_channels=80, channels=64, layers=10, stacks=2, quantization=256):
        super().__init__()

        # WaveNet typically uses niu-law quantization:

        # input shape: (batch, mel_channels, time)
        # mel spectrogram conditioning input

        self.input_conv = nn.Conv1d(mel_channels, channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for _ in range(stacks):
            for i in range(layers):
                dilation = 2 ** i
                self.blocks.append(ResidualBlock(channels, dilation))

        self.output1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.output2 = nn.Conv1d(channels, quantization, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, mel] -> [B, mel, T] for Conv1d
        x = self.input_conv(x)

        skip_connections = []

        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip) # save skips for later

        out = sum(skip_connections) # sum skips

        out = F.relu(out)
        out = F.relu(self.output1(out))
        out = self.output2(out)

        return out


class Tacotron2Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, voc_size=10000, num_layers=1):
        super().__init__()
        self.character_embedding = nn.Embedding(voc_size, input_size)
        self.conv1d_combo = nn.Sequential(
            nn.LazyConv1d(out_channels=hidden_size, kernel_size=5, padding=2),
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
        conv_out = self.conv1d_combo(embedded.transpose(1, 2)).transpose(1, 2)
        output, _ = self.bi_directional_lstm(conv_out)

        return output

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

        prev_mel = torch.zeros((batch_size, 1, self.mel_size), device=device)
        prev_attention = torch.zeros((batch_size, t_enc), device=device)
        lstm_hidden = None

        mel_outputs, stop_outputs, attention_weights = [], [], []

        num_steps = mel_targets.size(1) if mel_targets is not None else max_steps

        for t in range(num_steps):
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

                if (torch.sigmoid(stop_token) > 0.5).all():
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
        self.encoder = Tacotron2Encoder(input_size, encoder_out, num_layers=num_layers)
        self.decoder = Tacotron2Decoder(mel_size, prenet_hidden, encoder_out, decoder_lstm_hidden, num_layers)
        self.vocoder = WaveNet()

    def forward(self, x, mel_targets=None):
        encoder_outputs = self.encoder(x)
        mel_outputs_post, mel_outputs, stop_outputs, attention_weights = self.decoder(encoder_outputs, mel_targets)
        vocoder_outputs = self.vocoder(mel_outputs_post)

        return mel_outputs, mel_outputs_post, stop_outputs, attention_weights, vocoder_outputs

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()

        # Total Loss Formula
        # Loss = MSE(mel, target)(decoder) + MSE(mel_post, target)(postnet)
        # + BCE(gate, target)(stop_token)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, mel_out, mel_out_postnet, gate_out, mel_target, gate_target):
        # mel_out: [B, T_dec, 80]
        # mel_out_postnet: [B, T_dec, 80]
        # mel_target: [B, T_dec, 80]

        # gate_out: [B, T_dec]
        # gate_target: [B, T_dec]

        # Mel losses
        mel_loss = self.mse_loss(mel_out, mel_target)
        mel_post_loss = self.mse_loss(mel_out_postnet, mel_target)

        # Gate loss
        gate_loss = self.bce_loss(
            gate_out.view(-1, 1),
            gate_target.view(-1, 1)
        )

        # Total loss
        total_loss = mel_loss + mel_post_loss + gate_loss

        return total_loss, mel_loss, mel_post_loss, gate_loss

def get_tacotron2(input_size, mel_size, prenet_hidden, postnet_hidden):
    return Tacotron2(input_size, mel_size, prenet_hidden, postnet_hidden, decoder_lstm_hidden=1024, num_layers=2)

def load_tacotron2(path, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = get_tacotron2(input_size=80, mel_size=80, prenet_hidden=256, postnet_hidden=512)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def get_tacotron2_loss():
    return Tacotron2Loss()

def get_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_dataset(dataset_path: str, audio_dir_path: str):
    return LJSpeechDataset(dataset_path, audio_dir_path)

def prepare_dataloader(dataset_path: str, audio_dir_path: str, batch_size: int, shuffle: bool = True):
    dataset = get_dataset(dataset_path, audio_dir_path)
    return get_dataloader(dataset, batch_size, shuffle)

def train(model, dataloader, epochs, loss_fn, optimizer):
    model.to(device)
    model.train()

    total_loss = 0

    for epoch in range(epochs):

        print(f'Epoch [{epoch + 1}/{epochs}]')

        for batch_idx, (text, audio) in enumerate(dataloader):
            text = text.to(device)
            audio = audio.to(device)

            optimizer.zero_grad()
            # Transpose audio from [B, 80, T] to [B, T, 80] to match model output shape
            mel_target = audio.transpose(1, 2)
            mel_outputs, mel_outputs_post, stop_outputs, attention_weights, vocoder_outputs = model(text, mel_target)

            # Build gate target: 0 for all frames, 1 at the last frame (stop token)
            gate_target = torch.zeros(mel_target.size(0), mel_target.size(1), device=device)
            gate_target[:, -1] = 1.0

            loss, mel_loss, mel_post_loss, gate_loss = loss_fn(mel_outputs, mel_outputs_post, stop_outputs, mel_target, gate_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}'
            )

        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}'
        )

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'tacotron2_epoch_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')

    torch.save(model.state_dict(), 'tacotron2_final.pth')
    print('Model saved at final epoch')

if __name__ == '__main__':
    print("Starting training...")
    model = load_tacotron2("tacotron2_epoch_31.pth")
    loss_fn = get_tacotron2_loss()
    optimizer = get_optimizer(model, 1e-4)
    dataloader = prepare_dataloader(f'{PROJECT_ROOT}/data/ljspeech/LJSpeech-1.1/metadata.csv',
                                 f'{PROJECT_ROOT}/data/ljspeech/LJSpeech-1.1/wavs',
                                 32)
    train(model, dataloader, 100, loss_fn, optimizer)
