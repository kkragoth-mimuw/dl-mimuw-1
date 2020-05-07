import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

from models.net01 import Net01, model_name as net01_model_name
from models.net02 import Net02, model_name as net02_model_name
from models.net02_no_batchnorm import Net02NoBatchNorm, model_name as net02_no_batchnorm_model_name


# Model creation
BATCH_SIZE=256
net = Net02()
current_model = net02_model_name
criterion = nn.CrossEntropyLoss()
optimizer = None


model_id=f'{current_model}_batch_{BATCH_SIZE}_optimizer_sgd'
# 

def load_data(location, batch_size=1):
    transform = transforms.Compose([
        transforms.RandomCrop(246),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = ImageFolder(os.path.join(location, 'train'), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2
                                            )

    testset = ImageFolder(os.path.join(location, 'test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = os.listdir(os.path.join(location, 'train'))

    return trainset, trainloader, testset, testloader, classes


def confusion_matrix(preds, labels, n_classes):

    preds = torch.argmax(preds, 1)
    conf_matrix = torch.zeros(n_classes, n_classes)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1

    print(conf_matrix)
    TP = conf_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        sensitivity = (TP[c] / (TP[c]+FN))
        specificity = (TN / (TN+FP))

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))
        print('Sensitivity = {}'.format(sensitivity))
        print('Specificity = {}'.format(specificity))


def main():
    global optimizer
    
    print(f"Model id: {model_id}")

    with open("config.json") as f:
        config = json.load(f)

    model_file = os.path.join(config['MODEL_ROOT'], model_id)

    trainset, trainloader, testset, testloader, classes = load_data(config['DATA_FOLDER'], BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device {device}")

    lmbda = lambda epoch: 0.95
    
    if device != "cpu":
        net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    starting_epoch = 0
    accuracies_test = []
    accuracies_train = []
    losses_train = []
    losses_test = []
    times = []

    try:
        checkpoint = torch.load(model_file)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        accuracies_test = checkpoint['accuracies_test']
        accuracies_train = checkpoint['accuracies_train']
        losses_train = checkpoint['losses_train']
        losses_test = checkpoint['losses_test']
        times = checkpoint['times']
        print(f"Loaded saved model, starting at epoch: {starting_epoch}")
    except Exception as e:
        starting_epoch = 0
        accuracies_test = []
        accuracies_train = []
        losses_train = []
        losses_test = []
        times = []
        print("Unable to load model")
        print(e)

    for epoch in range(starting_epoch, config['EPOCHS']):  # loop over the dataset multiple times
        print(f"*Running epoch {epoch + 1}...")
        start_epoch = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            if device != "cpu":
                inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # running_loss += loss.item()
            # if i % 100 == 99:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        

        epoch_time = time.time() - start_epoch

        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = None, None
                if device == "cpu":
                    images, labels = data
                else:
                    images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train+= (predicted == labels).sum().item()

        accuracies_train.append(correct_train/total_train)

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = None, None
                if device == "cpu":
                    images, labels = data
                else:
                    images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test+= (predicted == labels).sum().item()

        accuracies_test.append(correct_test/total_test)
        times.append(epoch_time)

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracies_test': accuracies_test,
            'accuracies_train': accuracies_train,
            'losses_train': losses_train,
            'losses_test': losses_test,
            'times': times  
        }, model_file)

        print("epoch %d/%d" % (epoch+1, config['EPOCHS']))
        print("time per epoch: %s seconds" % (time.time() - start_epoch))
        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct_test / total_test))

    print('***Finished training. Summary: ***')
    print()
    print("Epoch | Accuracy Test | Accuracy Train")
    print("-----------------|")
    for i, (acc_test, acc_train) in enumerate(zip(accuracies_test, accuracies_train)):
        print(f' {i} | {acc_test} | {acc_train}')

    plt.figure()
    plt.title(f"{model_id} accuracy test")
    plt.axis([0, len(accuracies_test) - 1, 0, 1])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(accuracies_test)
    plt.savefig(os.path.join(config['PLOTS_FOLDER'], (model_id + "_accuracy_test")))

    plt.figure()
    plt.title(f"{model_id} accuracy train")
    plt.axis([0, len(accuracies_train) - 1, 0, 1])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(accuracies_train)
    plt.savefig(os.path.join(config['PLOTS_FOLDER'], (model_id + "_accuracy_train")))


    plt.figure()
    plt.title(f"{model_id} times")
    plt.xlabel('epoch')
    plt.ylabel('time')
    plt.axis([0, len(times) - 1, 0, 100])
    plt.plot(times)
    plt.savefig(os.path.join(config['PLOTS_FOLDER'], (model_id + "_times")))


    # Confusion matrix:
    conf_matrix = torch.zeros(len(classes), len(classes))

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    sensitivity = list(0. for i in range(len(classes)))
    specificity = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if device != "cpu":
                images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for p, t in zip(predicted, labels):
                conf_matrix[p, t] += 1

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        TP = conf_matrix.diag()
        for i in range(len(classes)):
            idx = torch.ones(len(classes)).byte()
            idx[i] = 0
            TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
            FP = conf_matrix[i, idx].sum()
            FN = conf_matrix[idx, i].sum()

            sensitivity = (TP[i] / (TP[i]+FN))
            specificity = (TN / (TN+FP))

            print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[i], TN, FP, FN))
            print('Sensitivity = {}'.format(sensitivity))
            print('Specificity = {}'.format(specificity))

        print(conf_matrix)

    

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    plt.figure()
    plt.title(f"{model_id} confusion_matrix")
    plt.imshow(conf_matrix, cmap='hot', interpolation='nearest')
    plt.savefig(os.path.join(config['PLOTS_FOLDER'], (model_id + "_confusion_matrix")))
            

if __name__ == "__main__":
    main()
