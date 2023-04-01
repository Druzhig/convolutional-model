import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# set the paths to the dataset folders
train_glass_dir = r'C:\Users\hp\Desktop\фото\датасет\train\стекло'
train_plastic_dir = r'C:\Users\hp\Desktop\фото\датасет\train\пластик'
validation_glass_dir = r'C:\Users\hp\Desktop\фото\датасет\valid\стекло'
validation_plastic_dir = r'C:\Users\hp\Desktop\фото\датасет\valid\пластик'

# define the data generators for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# create the data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    directory=r'C:\Users\hp\Desktop\фото\датасет\train',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=123
)

validation_generator = validation_datagen.flow_from_directory(
    directory=r'C:\Users\hp\Desktop\фото\датасет\valid',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# create the model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_generator, epochs=50, validation_data=validation_generator)

# save the model
model.save('C:\\Users\\hp\\Desktop\\фото\\модель_бутылки.h5')

# use the model for detection
# import cv2
#
# # load the model
# model = tf.keras.models.load_model('C:\\Users\\hp\\Desktop\\фото\\модель_бутылки.h5')
#
# # read the video
# cap = cv2.VideoCapture(0)
#
# # loop through each frame in the video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # resize the frame to the same size as the input image of the model
#     img = cv2.resize(frame, (224, 224))
#
#     # normalize the pixel values
#     img = img / 255.0
#
#     # add a batch dimension and predict the label using the model
#     label = model.predict(tf.expand_dims(img, axis=0))[0][0]
#
#     # display the label on the frame
#     if label > 0.5:
#         cv2.putText(frame, 'Plastic', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     else:
#         cv2.putText(frame, 'Glass', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

#имеется 2 папки с фотографиями. напиши код, который обучает модель, которая определяет по моей вебкамере в ноутбуке, стеклянная или пластиковая бутылка