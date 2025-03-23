import json

with open('./ImageNet1k_labels.json', 'r') as file:
        labels_1k = json.load(file)
with open('./data/imagenet100/Labels.json', 'r') as file:
        labels_100 = json.load(file)

real_labels = []

# for class_name in labels_100.values():
#     for i, name in enumerate(labels_1k.values()):
#         if name == class_name:
#             real_labels.append(i)
#             break
# print(len(real_labels))

# with open('./data/imagenet100/Real_Labels.json', 'w') as file:
#          json.dump(real_labels,file)


import os  # import os module

directory = './data/imagenet100/val_sub'  # set directory path
real_class_indexes = []
for name in sorted(os.listdir(directory)):
    label = labels_100[name]
    
    for i, class_label in enumerate(labels_1k.values()):
        if class_label == label:
            real_class_indexes.append(i)
            break

with open('./data/imagenet100/Real_Labels.json', 'w') as file:
         json.dump(real_class_indexes,file)       



    