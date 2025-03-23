import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from config import cfg
import matplotlib.pyplot as plt

dir = "preact_k256_e32_n100_adv"

for file in os.listdir(fr"./logs/preact_k256_e32_ni100_adv"):
    print(file)
    if file.startswith("events.out.tfevents.1742040157"):
        filepath = os.path.join(dir, file)
        print(f"Reading: {filepath}")
        ea = EventAccumulator(filepath)
        ea.Reload()
        # available tags
        scalar_tags = ea.Tags().get('scalars', [])
        print("Available scalar tags:", scalar_tags)
        
        tag = 'loss_epoch'

        if tag in scalar_tags:
            print(f"tag: {tag}")
            epochs = list()
            values = list()
            for event in ea.Scalars(tag):
                print(f"Step: {event.step}, Value: {event.value}")
                epochs.append(event.step)
                values.append(event.value)
            plt.plot(epochs, values)
            plt.title(tag)
            plt.show()
        else:
            print("no tag")


            
