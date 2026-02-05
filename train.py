import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import get_model, BinaryConv2d, BinaryLinear
import argparse
import os
import torch.nn.functional as F

def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Latent weight clipping for binarized layers
        # This is critical for BNN training to keep weights responsive
        for m in model.modules():
            if hasattr(m, 'weight') and isinstance(m, (BinaryConv2d, BinaryLinear)):
                m.weight.data.clamp_(-1, 1)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 50 == 0:
            print(f'Batch [{i+1}/{len(trainloader)}] | Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%', flush=True)
    
    return running_loss / len(trainloader), 100. * correct / total

def test(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(testloader), 100. * correct / total

def train_kd(student, teacher, trainloader, criterion, optimizer, device, temperature=3.0, alpha=0.5):
    student.train()
    teacher.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        
        # Student and Teacher outputs
        student_outputs = student(inputs)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)
        
        # Soft targets loss (KD)
        loss_ce = criterion(student_outputs, labels)
        loss_kd = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1)
        ) * (temperature ** 2)
        
        loss = alpha * loss_ce + (1 - alpha) * loss_kd
        
        loss.backward()
        optimizer.step()
        
        # Latent weight clipping
        for m in student.modules():
            if hasattr(m, 'weight') and isinstance(m, (BinaryConv2d, BinaryLinear)):
                m.weight.data.clamp_(-1, 1)
        
        running_loss += loss.item()
        _, predicted = student_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 50 == 0:
            print(f'Batch [{i+1}/{len(trainloader)}] | Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%', flush=True)
    
    return running_loss / len(trainloader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--model', default='xnor', type=str, help='model type (xnor or baseline)')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--kd', action='store_true', help='use knowledge distillation')
    parser.add_argument('--teacher-path', type=str, help='path to teacher model checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    student = get_model(args.model).to(device)
    teacher = None
    if args.kd:
        print(f"Loading teacher model from {args.teacher_path}...")
        teacher = get_model('baseline').to(device)
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
        teacher.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        if args.kd and teacher:
            train_loss, train_acc = train_kd(student, teacher, trainloader, criterion, optimizer, device)
        else:
            train_loss, train_acc = train(student, trainloader, criterion, optimizer, device)
            
        test_loss, test_acc = test(student, testloader, criterion, device)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        print('-' * 20)

    # Save model
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    suffix = '_kd' if args.kd else ''
    torch.save(student.state_dict(), f'checkpoints/{args.model}{suffix}_cifar10.pth')

if __name__ == '__main__':
    main()
