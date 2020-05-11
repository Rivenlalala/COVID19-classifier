


transform1 = CustomCompose([transforms.Resize((224, 224))])
transform3 = CustomCompose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()])

transform2 = transforms.Compose([transforms.Resize((224, 224)),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomRotation(10),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5])])
data = datasets.ImageFolder(root='dataset/train', transform=transform3)
raw_data = np.array([np.array(image[0], dtype="float") for image in data])
raw_minority_data = np.array([np.array(image[0], dtype="float") for image in data if image[1]==0])
test_data = datasets.ImageFolder(root='dataset/test', transform=transform1)
transform_smote = transforms.Compose([transforms.Resize((224, 224)),
                                      Smote(raw_minority_data, 5),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                           std=[0.5, 0.5, 0.5])])
transform_SP_minority = transforms.Compose([transforms.Resize((224, 224)),
                                   SampleParing(raw_data, minority_only=True),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])])
transform_SP = transforms.Compose([transforms.Resize((224, 224)),
                                   SampleParing(raw_data, minority_only=False),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])])
transform_RICAP = transforms.Compose([transforms.Resize((224, 224)),
                                   RICAP(data, beta=1),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])])
transform_Maj = transforms.Compose([transforms.Resize((224, 224)),
                                   Majority(raw_data),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])])
data_smote = datasets.ImageFolder(root='dataset/train', transform=transform_smote)
data_SP_minority = datasets.ImageFolder(root='dataset/train', transform=transform_SP_minority)
data_SP = datasets.ImageFolder(root='dataset/train', transform=transform_SP)
data_RICAP = datasets.ImageFolder(root='dataset/train', transform=transform_RICAP)
data_Maj = datasets.ImageFolder(root='dataset/train', transform=transform_Maj)
data = datasets.ImageFolder(root='dataset/train', transform=transform2)

model = DenseNet121().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#data = datasets.ImageFolder(root='dataset/train', transform=transform2)
dataloader = DataLoader(data_smote, batch_size=16, shuffle=True)


from random import random
class Smote():

    def __init__(self, raw_dataset, k):
        self.dataset = raw_dataset
        self.k = k

    def knn(self, img):
        img = img.reshape(-1)#
        data = self.dataset.reshape(len(self.dataset), -1)#二维化一维？
        distance = np.linalg.norm(img - data, axis=1)#array差的2次基
        nearest = np.argsort(distance)#把第distance中第一小到第n小的序号列出

        return self.dataset[nearest[:self.k]]#差值前k小的img和data的差

    def __call__(self, sample):
        print(sample.shape)
        if (sample[1] == 1 | (random() > 0.5)):
            return sample
        else:
            img = np.array(sample[0], dtype="float64")
            new = img
            nearests = self.knn(img)
            for nearest in nearests:
                new += random() * (nearest - img)#new 最终是标差与img的差乘以随机数的和
            #new = (new - new.min()) / (new.max() - new.min()) * 255
            new = Image.fromarray(np.uint8(new))
            return {new, 0}

transform1 = transforms.Compose([transforms.Resize((224, 224))])
transform3 = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()])
data = datasets.ImageFolder(root='dataset/train', transform=transform1)
raw_minority_data = np.array([np.array(image[0], dtype="float") for image in data if image[1]==0])

transform_smote = transforms.Compose([transforms.Resize((224, 224)),
                                      Smote(raw_minority_data, 5),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                           std=[0.5, 0.5, 0.5])])
data = datasets.ImageFolder(root='dataset/train', transform=transform3)
raw_minority_data = np.array([np.array(image[0], dtype="float") for image in data if image[1]==0])
data_smote = datasets.ImageFolder(root='dataset/train', transform=transform_smote)
dataloader = DataLoader(data_smote, batch_size=1, shuffle=True)


for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
      inputs, labels = data
      inputs = inputs.cuda()
      labels = labels.cuda()
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 20 == 19:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 20))
        print(outputs)
        running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'model.pth')



TP = 0
TN = 0
FP = 0
FN = 0
model = model.eval()
with torch.no_grad():
    for data in dataloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        predicted = torch.argmax(outputs, 1)
        for i, pred in enumerate(predicted):
            if pred == labels[i] & pred == 1:
                TN += 1
            elif pred == labels[i] & pred == 0:
                TP += 1
            elif pred != labels[i] & pred == 0:
                FN += 1
            else:
                FP += 1

print(TN, TP, FN, FP)


from torch.utils.data import Dataset, DataLoader
transform3 = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor()])
test_data = datasets.ImageFolder(root='dataset/test', transform=transform3)

testloader = DataLoader(test_data, batch_size=16, shuffle=False)



TP = 0
TN = 0
FP = 0
FN = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        predicted = torch.argmax(outputs, 1)
        for i, pred in enumerate(predicted):
            if pred == labels[i] & pred == 1:
                TN += 1
            elif pred == labels[i] & pred == 0:
                TP += 1
            elif pred != labels[i] & pred == 0:
                FN += 1
            elif pred != labels[i] & pred == 1:
                FP += 1

print(TN, TP, FN, FP)