```python
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        mask_name = os.path.join(self.root_dir, self.image_files[idx].replace('.jpg', '_mask.png'))

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
```


```python
import torch
import torchvision

def create_model(n_classes):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    return model
```


```python
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        print('Loss: {:.4f}'.format(epoch_loss))

    return model
```


```python
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                     std = [0.229, 0.224, 0.225])])

dataset = SegmentationDataset(root_dir='dataset/', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_model(n_classes = 21)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

model = train_model(model, dataloader, criterion, optimizer, num_epochs=25)
```


```python
from skimage.measure import label

def semantic_to_instance(segmentation):
    labels = label(segmentation)
    return labels
```

For the second step, we will use `skimage.morphology.convex_hull_image` to create a convex hull for each instance.

```python
from skimage.morphology import convex_hull_image

def instance_to_convex_hull(labels):
    hulls = np.zeros_like(labels)
    for i in np.unique(labels)[1:]:  # skip the background
        hulls[labels == i] = convex_hull_image(labels == i)
    return hulls
```

Finally, we can find the orientated bounding box of the convex hull, which is a rough approximation of finding the minimal bounding polygon with 4 vertices.

```python
from skimage.measure import regionprops

def get_bounding_box(hulls):
    props = regionprops(hulls)
    boxes = [prop.bbox for prop in props]
    return boxes
```

You can use these helper functions together like:

```python
# Assuming our output semantic segmentation is in variable output, 
output = model(inputs)['out']
output = torch.argmax(output, dim=0)

# Convex Hull for each segmentation
instances = semantic_to_instance(output.cpu().numpy())
convex_hulls = instance_to_convex_hull(instances)

# Getting bounding boxes
bounding_boxes = get_bounding_box(convex_hulls)
```