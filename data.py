# Imports
import os
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import pandas as pd
import spacy
# Spacy eng
spacy_eng = spacy.load("en_core_web_sm")
from PIL import Image
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence

# Data paths
DATA_PATH = "/home/moez/Desktop/image_annotation/flickr8k"
caption_file = DATA_PATH + "/captions.txt"
images_directory = "/Flickr8k_Dataset"

# Creating Vocabulary
class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]  

class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)

# Transformations
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
SIZE_IN_PIXELS = (INPUT_HEIGHT, INPUT_WIDTH)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Data Augmentation
transformations = T.Compose([
    T.Resize(SIZE_IN_PIXELS),
    T.RandomRotation(15),
    T.RandomCrop(224, padding=2),
    T.ColorJitter(),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)])

# Creating dataset
dataset = FlickrDataset(
    root_dir = DATA_PATH+images_directory,
    captions_file = DATA_PATH+caption_file,
    transform=transformations)

def get_dataset():
    return dataset

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets

def get_data_loader(batch_size=128):
    BATCH_SIZE = batch_size
    NUM_WORKER = 4
    #token to represent the padding
    pad_idx = dataset.vocab.stoi["<PAD>"]
    print("Creating dataloader.")
    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        pin_memory=True,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )

print("Dataloader created.")