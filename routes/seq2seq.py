from flask import Blueprint, render_template, session, current_app
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

seq2seq_bp = Blueprint('seq2seq', __name__)

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if x.size(-1) != self.input_size:
            raise ValueError(f"Input feature size mismatch. Got {x.size(-1)}, expected {self.input_size}")
        
        # Encoder
        _, (hidden, cell) = self.encoder(x)
        
        # Prepare decoder input (use last encoder output as first decoder input)
        decoder_input = x[:, -1:, :]  # Shape: [batch, 1, input_size]
        
        # Decoder
        output, _ = self.decoder(decoder_input, (hidden, cell))
        return self.fc(output)

class AyatDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

@seq2seq_bp.route('/analysis')
@seq2seq_bp.route('/analysis')
def seq2seq_analysis():
    try:
        # 1. Load data
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], session['uploaded_file'])
        df = pd.read_csv(filepath)
        text = " ".join(df['Teks'].astype(str).tolist())  # Pastikan semua teks string
        
        # 2. Bangun vocabulary LENGKAP dulu
        chars = ['<PAD>', '<UNK>'] + sorted(list(set(text)))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        vocab_size = len(chars)  # Ini akan menentukan input_size
        print(f"Vocabulary size: {vocab_size}")  # Debugging
        
        # 3. Baru inisialisasi model dengan vocab_size yang benar
        hidden_size = 64  # Bisa diubah, tapi input_size harus = vocab_size
        model = Seq2Seq(input_size=vocab_size, 
                       hidden_size=hidden_size, 
                       output_size=vocab_size)
        
        # 4. Fungsi pembuat tensor
        def create_tensor(sequence, vocab_size):
            return torch.FloatTensor(np.eye(vocab_size)[sequence]).unsqueeze(0)  # [1, seq_len, vocab_size]
        
        # 5. Persiapan data training
        seq_length = 20
        sequences = []
        for i in range(0, len(text) - seq_length, 1):
            seq_in = text[i:i+seq_length]
            seq_out = text[i+1:i+seq_length+1]
            sequences.append((
                [char_to_idx.get(c, 1) for c in seq_in],  # 1 untuk <UNK>
                [char_to_idx.get(c, 1) for c in seq_out]
            ))
        
        # 6. Validasi dimensi sebelum training
        test_seq = sequences[0][0]
        test_input = create_tensor(test_seq, vocab_size)
        print(f"Test input shape: {test_input.shape}")
        assert test_input.shape[2] == vocab_size, "Dimensi input tidak sesuai!"
        
        # 7. Training loop (contoh singkat)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        losses = []  # List to store loss values
        
        for epoch in range(3):
            for seq_in, seq_out in sequences[:50]:  # Gunakan subset dulu
                inputs = create_tensor(seq_in, vocab_size)
                targets = torch.LongTensor(seq_out)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, vocab_size), targets)
                losses.append(loss.item())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                        
        # 6. Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Training Loss")
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # 7. Prediction
        test_phrase = "Bimbinglah kami"[:seq_length]
        test_input = create_tensor([char_to_idx.get(c, 1) for c in test_phrase], vocab_size)
        
        with torch.no_grad():
            output = model(test_input)
        predicted_ids = torch.argmax(output, dim=-1).squeeze().numpy()
        predicted_text = ''.join([idx_to_char.get(idx, '') for idx in predicted_ids])
        
        return render_template('seq2seq.html',
                            plot_url=plot_url,
                            original_text=test_phrase,
                            predicted_text=predicted_text,
                            sample_data=df.sample(3).to_dict('records'))
    
    except Exception as e:
        current_app.logger.error(f"Error in seq2seq: {str(e)}", exc_info=True)
        return render_template('seq2seq.html', 
                            error=f"Terjadi error: {str(e)}. Pastikan data cukup dan format sesuai.")