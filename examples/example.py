import sys, os
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src_path)


from imagedataset import AdvanceImageFolder
from torchvision.transforms import ToTensor, Resize, Compose


transform = Compose([Resize([224, 224]), ToTensor()])

dataset = AdvanceImageFolder(root="example_dataset",
                             name="my_great_dataset",
                             transform=transform)

# ..to load full dataset to RAM
dataset.load_ram(verbose=True)

# split the dataset
d_train, d_test = dataset.random_split(percentages=[0.75, 0.25], 
                                       split_names=["train", "test"])

print(d_train)
print(d_test)

# get dataloaders
loader_train = d_train.dataloader(batch_size=2, drop_last=False, shuffle=True)
loader_test  = d_test.dataloader(batch_size=2, drop_last=False, shuffle=True)


# when updating pseudolabels they are updated in the dataset, but not in the loader, so
# we need to reitarete or reinitialize the dataloader to get updated pseudolabels.

# For example

print()
print("First Run")
for batch in loader_train:
    index = batch.index
    id = batch.id
    pseudolabel = batch.pseudolabel
    label = batch.label 

    print(f"BEFORE: Index: {index}, id: {id}, pseudolabel: {pseudolabel}, label: {label}")
    d_train.update_pseudolabels(values=index, indices=index)
    print(f"AFTER: Index: {index}, id: {id}, pseudolabel: {pseudolabel}, label: {label}") 
    # same as before
    print()


print("Second Run")
for batch in loader_train:
    index = batch.index
    id = batch.id
    pseudolabel = batch.pseudolabel
    label = batch.label 

    print(f"Index: {index}, id: {id}, pseudolabel: {pseudolabel}, label: {label}")
    # now they are updated