from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomZoom, RandomRotation

# THIS IS A STUPID WAY OF DOING THIS BUT I DON'T CARE
model_name = input("Enter model name: ")
model = Sequential()
# INSERT MODEL HERE, REMEMBER input_shape= ... ...(128, 128, 3) ... ... 
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# INSERT MODEL ABOVE, REMEMBER input_shape= ... ...(128, 128, 3) ... ...
plot_model(model, to_file=f'./documentation/images/{model_name}.png', show_shapes=True, show_layer_names=True,dpi=300)

