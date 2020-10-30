import json
import pickle
import torch
from pytorch_pretrained_bert_inset import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased', state_dict=torch.load('models/BERT-pretrain-1-step-5000.pkl')).cuda()

data = pickle.load(open('dataset/tripadvisor_review_processed_uncut.json', 'rb'))
print('total number of paragraphs is', len(data))

cut = []
paragraphs = []
model_bert.eval()
with torch.no_grad():
    for k in tqdm.tqdm(range(len(data))):
        if len(data[k]) > 6:
            ids_unpad = []
            for i in range(len(data[k])):
                ids_unpad.append(torch.tensor([101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data[k][i])) + [102], dtype=torch.long))
            ids = pad_sequence(ids_unpad, batch_first=True, padding_value=0).cuda()
            x = torch.zeros(ids.size(0), 768).cuda()
            if ids.size(-1) < 32:
                encoded_layers, _ = model_bert(ids, torch.zeros_like(ids), 1 - (torch.zeros_like(ids) == ids).type(torch.uint8), False)
                x[:, :] = encoded_layers[:, 0, :]
                paragraphs.append(x.half())
                cut.append(data[k])
        torch.cuda.empty_cache()

print(len(paragraphs), ' out of ', k)
torch.save(paragraphs, 'dataset/trip_cut_half.pt')
json.dump(cut, open('dataset/tripadvisor_review_processed_cut.json', 'w'))
