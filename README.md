# torch image dataset

## Datasets for image classification

### AdvanceImageFolder
**Use case**: 
- keep track of pseudolabels (read/update).
- load full dataset into RAM to speed up training when disk memory access is slow.
- easy subsetting/splitting keeping the possibility to update and read pseudolabels.

It is a subclass of ``` torchvision.datasets.ImageFolder``` so it asks for a dataset structure in the form:
```
  root\
    class1\
      img1
      img2
      ...
    class2\
      img1
      img2
      ...
    ...   
```

The dataset is initialized by indicating the root directory.
As ImageFolder it is possible to indicate the transorm for inputs and for targets. Additionally, the name of the dataset can be indicated and it is possible to load just a subset of the dataset. ```load_percentage``` specifies the percentage of images to load (randomly picked without repetitions) or ```indices``` can be used to select specific indices of the dataset.

```
  dataset = AdvanceImageFolder(root="path/to/mydataset/",
                               name="incredible_dataset",
                               transform=None,
                               target_transform=None,
                               load_percentage=1.,
                               indices=None)
```

The return type of the dataset is ```types.SimpleNamespace``` with the following fields:
- image
- id
- index
- label
- pseudolabel

For example you can:

```
  item = dataset.__getitem__(0)
  image = item.image
  label = item.label
  pseudolabel = item.pseudolabel
  ...
```

To use a dataloader it is possible to use:
```
  loader = dataset.dataloader(batch_size=10)
```
Actually the dataloader function accepts the same arguments as ```torch.utils.data.DataLoader``` and returns a proper ```DataLoader``` with the appropriate ```collate_fn``` for batching.

To subset a dataset:
```
    my_subset = dataset.subset(indices=[0,1,2], ids=None)
```
It is possible to select the indices or the ids of images. The indices are the positions of images in the sequence from 0 to len(dataset)-1.
Ids are initialized as indices but they are kept constant while subsetting and splitting and, for this reason, it is possible to get the index of an image in the original dataset.

To split a dataset:

```
    train_d, val_d, test_d = dataset.random_split(percentages=[0.5, 0.3, 0.2])
```

To drop a random part of the dataset, let say we want to keep just 50% of dataset:
```
    dataset = datset.random_split(percentages=[0.5])
```  

To load the dataset into RAM just use:
```
  dataset.load_ram()
```
And if you want to release the allocated memory while keeping the dataset:
```
  dataset.clean_ram()
```




