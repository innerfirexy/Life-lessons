from models import TransformerModel, MixedTransformerModel, RNN, RNN_padded, MixedRNN
import torch
import os

model_paths = ['best_lstm_gesture_gesture.pt',\
               'best_lstm_gesture_gesture_compressed.pt',\
               'best_trm_gesture_gesture.pt',\
               'best_trm_gesture_gesture_compressed.pt',\
               'best_trm_mix_mix.pt']

for model_path in model_paths:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    output_path, _ = os.path.splitext(model_path)
    output_path = output_path + '_state_dict.pth'
    torch.save(model.state_dict(), output_path)
