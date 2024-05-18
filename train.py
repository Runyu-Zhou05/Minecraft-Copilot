import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
import os
from PIL import Image
from tqdm import tqdm

def seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)

class Set(Dataset):
    def __init__(self, imgs, labels):
        super().__init__()
        self.X = imgs
        self.y = labels
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def make_dataloader(batch_size, ratio):
    try:
        imgs = torch.load('imgs.pt')
        labels = torch.load('labels.pt')

    except:
        pos_name = os.listdir('1')
        neg_name = os.listdir('0')
        imgs = torch.tensor([])
        labels = []

        for name in tqdm(pos_name):
            img = Image.open('1/' + name)
            img = transforms.Resize((224, 224))(img)
            img = transforms.ToTensor()(img)
            imgs = torch.cat([imgs, img.unsqueeze(0)], dim=0)
            labels.append(1)
        
        for name in tqdm(neg_name):
            img = Image.open('0/' + name)
            img = transforms.Resize((224, 224))(img)
            img = transforms.ToTensor()(img)
            imgs = torch.cat([imgs, img.unsqueeze(0)], dim=0)
            labels.append(0)
        
        labels = torch.tensor(labels)
        torch.save(imgs, 'imgs.pt')
        torch.save(labels, 'labels.pt')
    
    all_set = Set(imgs, labels)
    train_size = int(ratio*len(all_set))
    train_set, test_set = random_split(all_set, [train_size, len(all_set)-train_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_components(device, lr, wd):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss().to(device)
    return model, opt, criterion

def train(datas, components, epochs):
    train_loader, test_loader = datas['train_loader'], datas['test_loader']
    model, opt, criterion, device = components['model'], components['opt'], components['criterion'], components['device']
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_acc = 0
        total_train_loss = 0
        total_train_size = 0

        for img, label in tqdm(train_loader):
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            opt.step()
            opt.zero_grad()

            pred = out.argmax(1)
            right = (pred == label).sum().item()
            size = len(label)

            total_train_acc += right
            total_train_loss += loss.item() * size
            total_train_size += size

        total_train_acc /= total_train_size
        total_train_loss /= total_train_size

        model.eval()
        total_test_acc = 0
        total_test_loss = 0
        total_test_size = 0
        
        with torch.no_grad():
            for img, label in tqdm(test_loader):
                img = img.to(device)
                label = label.to(device)

                out = model(img)
                loss = criterion(out, label)

                pred = out.argmax(1)
                right = (pred == label).sum().item()
                size = len(label)

                total_test_acc += right
                total_test_loss += loss.item() * size
                total_test_size += size
            
            total_test_acc /= total_test_size
            total_test_loss /= total_test_size

        if total_test_acc > best_acc:
            best_acc = total_test_acc
            torch.save(model.state_dict(), 'best_model.pt')
        print(f"Epoch {epoch+1} - Train Loss: {total_train_loss:.4f}, Train Acc: {total_train_acc:.4f}"
              f" | Val. Loss: {total_test_loss:.4f}, Val. Acc: {total_test_acc:.4f}")

def main():
    seed(42)
    device = 'cuda'
    epochs = 10
    batch_size = 32
    lr = 1e-4
    wd = 1e-3
    ratio = 0.7

    train_loader, test_loader = make_dataloader(batch_size, ratio)
    datas = {'train_loader': train_loader, 'test_loader': test_loader}
    print("Data prepared")

    model, opt, criterion = get_components(device, lr, wd)
    components = {'model': model, 'opt': opt, 'criterion': criterion, 'device': device}
    print("Training prepared")
    train(datas, components, epochs)

if __name__ == '__main__':
    main()