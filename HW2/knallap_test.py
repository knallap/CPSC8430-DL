import sys
import torch
import json
from knallap_train import TestData, test, MODELS, EncoderRNN, DecoderRNN, Attention
from torch.utils.data import DataLoader
from data.bleu_eval import BLEU
import pickle

print("Loading model...")
model = torch.load('knallap_model.h5', map_location=lambda storage, loc: storage)
filepath = 'data/testing_data/feat'
dataset = TestData(filepath)
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print("Running inference on the test set...")
ss = test(testing_loader, model, i2w)
output_file = sys.argv[2]
with open(output_file, 'w') as f:
    for video_id, sentence in ss:
        f.write(f'{video_id},{sentence}\n')
print("Calculating BLEU scores...")
test_labels = json.load(open('data/testing_label.json'))
output = sys.argv[2]
result = {}
with open(output, 'r') as f:
    for line in f:
        line = line.strip()
        comma_pos = line.index(',')
        test_id = line[:comma_pos]
        caption = line[comma_pos + 1:]
        result[test_id] = caption

bleu_scores = []
for item in test_labels:
    captions = [x.rstrip('.') for x in item['caption']]  # Remove trailing periods from captions
    predicted_caption = result.get(item['id'], '')
    score = BLEU(predicted_caption, captions, True)
    bleu_scores.append(score)

average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score: {average_bleu:.4f}")
