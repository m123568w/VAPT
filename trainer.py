import PIL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import numpy as np

import clip

from data import ImageDataset
from net import MyKnowledgeNet

print(torch.__version__)
print(torch.version.cuda)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_labels = ["aircraft carrier.", 
                        "destroyer.", 
                        "cruiser.", 
                        "supply ship.", 
                        "cruise ship."]

    clip.available_models()

    model, preprocess = clip.load("/hy-tmp/clip_model/ViT-B-32.pt", device)
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    img_dataset = ImageDataset("./ship_dataset/")

    # 计算划分比例
    dataset_size = len(img_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # 固定随机种子保证可重复性（可选）
    generator = torch.Generator()

    # 划分数据集
    train_dataset, val_dataset = random_split(
        img_dataset,
        [train_size, val_size],
        generator=generator
    )

    # 验证划分结果
    print(f"总样本数: {dataset_size}")
    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=96, shuffle=True, drop_last=False, num_workers=32)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=96, shuffle=False, drop_last=False, num_workers=32)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=96, shuffle=True, drop_last=False, num_workers=32)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=96, shuffle=False, drop_last=False, num_workers=32)
    mymodel = MyKnowledgeNet(cls_labels, model, img_dataset.num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # backbone_params = list(map(id, mymodel.vt.parameters()))
    # align_parmas = filter(lambda p: id(p) not in backbone_params, mymodel.parameters())

    optimizer = optim.Adam([{'params': mymodel.self_attention.parameters()},
                        {'params': mymodel.linear.parameters()}], lr=1e-3)

    # 训练模型
    for epoch in range(30):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.to(device)
            with torch.no_grad():
                images = [preprocess(PIL.Image.open(image)) for image in inputs]
                image_input = torch.tensor(np.stack(images)).half().to(device)

            optimizer.zero_grad()  # move zero_grad before the forward pass
            output = mymodel(image_input)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(data), len(train_loader.dataset), 100. * i / len(train_loader), loss.item()))
                
                
    print('Finished Training')
    
    
    # 测试模型
    top_1_correct = 0
    top_2_correct = 0
    top_3_correct = 0
    total = 0
    # set the model to evaluation mode
    mymodel.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.to(device)
            images = [preprocess(PIL.Image.open(image)) for image in inputs]
            image_input = torch.tensor(np.stack(images)).half().to(device)
            
            outputs = mymodel(image_input)
            _, predicted = torch.topk(outputs.data, k=3, dim=1)
            total += labels.size(0)
            top_1_correct += (predicted[:, 0] == labels).sum().item()
            top_2_correct += ((predicted[:, 0] == labels) | (predicted[:, 1] == labels)).sum().item()
            top_3_correct += ((predicted[:, 0] == labels) | (predicted[:, 1] == labels) | (predicted[:, 2] == labels)).sum().item()

    print('Top-1 accuracy of the network on the %d test images: %.2f %%' % (len(test_loader.dataset), 100 * top_1_correct / total))
    print('Top-2 accuracy of the network on the %d test images: %.2f %%' % (len(test_loader.dataset), 100 * top_2_correct / total))
    print('Top-3 accuracy of the network on the %d test images: %.2f %%' % (len(test_loader.dataset), 100 * top_3_correct / total))
