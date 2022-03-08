import time

from . import LOADERS
from imagedataset import BasicImageFolder

def benchmark_loaders(path, epochs=10, verbose=True):

    results = {}

    for loader in LOADERS:
        dataset = BasicImageFolder(path, image_loader=loader, image_loader_resize=(224, 224))
        
        start = time.time()
        for _ in range(epochs):
            for i in range(len(dataset)):
                dataset.__getitem__(i)
        end = time.time()
        total_time = end - start
        
        results[loader] = total_time

        if verbose:
            print(f"Loader {loader}: {total_time:.5f}s")

def benchmark_dataloaders(path, epochs=10, cuda=True, batch_size=100, num_workers=8, loader="pil"):
    dataset = BasicImageFolder(path, image_loader=loader, image_loader_resize=(224, 224))
    dataloader = dataset.dataloader(cuda=cuda, batch_size=batch_size, num_workers=num_workers)

    start = time.time()
    for _ in range(epochs):
        for x in dataloader:
            image, label = x.image, x.label

    end = time.time()
    print(f"Loader {loader}: {end-start:.5f}s")