# torch image dataset
Install:
```
  pip install torch-image-dataset
```

## Requirements:
```
  pip install torch torchvision pillow opencv-python
```

### Additional requirements (optional)

For saving/loading the dataset in a database:
```
  conda install -c conda-forge python-lmdb
```

For fast image loaders:
```
  conda install -c conda-forge pyturbojpeg
  conda install -c conda-forge accimage
```

### Image Loaders
ImageLoaders are objects that can load an image from a path and decode it (additionally they can resize the image).
The image loaders available are: ```LoaderPIL``` (from pillow), ```LoaderOpenCV``` (from opencv), ```LoaderAccImage``` (from accimage) and ```LoaderTurboJPEG``` (from turbojpeg).
To initialize a loader:

```
  from imagedataset import LoaderOpenCV, Interpolation
  
  size = (224, 224)
  loader = LoaderOpenCV(size=size, interpolation=Interpolation.CV_AREA)
```

If the size is ```None``` there won't be any resize at loading time. The interpolation methods available are:

```
    CV_BILINEAR         (cv2.INTER_LINEAR)
    CV_BILINEAR_EXACT   (cv2.INTER_LINEAR_EXACT)
    CV_NEAREST          (cv2.INTER_NEAREST)
    CV_BICUBIC          (cv2.INTER_CUBIC)
    CV_LANCZOS          (cv2.INTER_LANCZOS4)
    CV_AREA             (cv2.INTER_AREA)

    PIL_NEAREST         (PIL Image.NEAREST)
    PIL_BICUBIC         (PIL Image.BICUBIC)
    PIL_BILINEAR        (PIL Image.BILINEAR
    PIL_LANCZOS         (PIL Image.LANCZOS)
    ACCIMAGE_BUILDIN    (interpolation of AccImage)
```
Not all loaders support all interpolations, but the opencv interpolations are accepted by all loaders (for example it is possible to use the CV_BILINEAR interpolation even in a LoaderPIL).

### Image Decoders
ImageDecoders are very similar to ImageLoaders but their ```__call__``` function accepts a ```bytes``` object and decodes to an image (and eventually resize it).
There is one decoder for each loader introduced before.
To create a decoder:
```
  decoder = DecoderPIL(size=(224,224), interpolation=Interpolation.CV_BICUBIC)
```

### Datasets

### AdvanceImageFolder
**Use case**: 
- keep track of pseudolabels (read/update).
- load full dataset into RAM to speed up training when disk memory access is slow.
- save and read the full dataset into a database to speed up training when the whole dataset cannot be stored in RAM.
- easy subsetting/splitting keeping the possibility to update and read pseudolabels.

Import:
```
  import imagedataset as id
```

The folder structure required is similar to the one of ``` torchvision.datasets.ImageFolder```:
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
Additionally, the name of the dataset can be indicated and it is possible to load just a subset of the dataset. ```load_percentage``` specifies the percentage of images to load (randomly picked without repetitions) or ```indices``` can be used to select specific indices of the dataset.

```
  dataset = id.AdvanceImageFolder(root="path/to/mydataset/",
                                  name="incredible_dataset",
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

**Note**: every tensor in the batch is of type int64 (classification labels) or uint8 (images or label arrays).

To speed up the training it is possible to use a ```CudaLoader``` that pre-fatches the samples:

```
  loader = dataset.cudaloader(batch_size=10)
```
**Note**: the CudaLoader convers images to floats but do not scale them, so they are still in range [0,255]. Additionally the CudaLoader can perform normalization, for instance to bring images in range [0,1] it is possible:
```
  loader = dataset.cudaloader(batch_size=10, mean=(0,0,0), std=(255, 255, 255))
```

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
To write the dataset into a LMDB database:
```
  dataset.write_database("mydb/path/")
```

Then, to get a dataset that reads the database:
```
  dataset = id.from_database("mydb/path/")
```







