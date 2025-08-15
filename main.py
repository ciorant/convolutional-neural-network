import torch
from oop_cnn import Convolutional
from data import train_data

image,_ = train_data[0]
channels = image.shape[0]
image_size = (image.shape[1],image.shape[2])
num_classes = len(torch.unique(train_data.targets))
hidden_units=10

model = Convolutional(input_channels=channels,
                   hidden_units=hidden_units,
                   num_classes=num_classes,
                   image_size=image_size)

model.setup_training(optimizer=torch.optim.Adam,
                     learning_rate=0.01)

from data import train_dataloader, val_dataloader, test_dataloader

model.fit(train_dataloader=train_dataloader,verbose=True,
          val_dataloader=val_dataloader, epochs=3)

X_pred, y_pred = model.predict(dataloader=test_dataloader)
model.save_model("cnn.path")
